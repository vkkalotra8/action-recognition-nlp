from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


try:
    # Used when imported as part of the scripts package.
    from .explanation_engine import (
        build_explanation_metadata,
        format_explanation_report,
    )

    from .predict_video_semantic import (
        CLASS_EMBEDDINGS_PATH,
        CLASS_NAMES_PATH,
        SCALER_PATH,
        VIDEO_EXTENSIONS,
        extract_sampled_video_frames,
        extract_visual_feature,
        find_frame_files,
        load_feature_extractor,
        load_json,
        load_projection_model,
        predict_semantic_class,
    )

except ImportError:
    # Used when executed directly:
    # python scripts/predict_video_explainable.py
    from explanation_engine import (
        build_explanation_metadata,
        format_explanation_report,
    )

    from predict_video_semantic import (
        CLASS_EMBEDDINGS_PATH,
        CLASS_NAMES_PATH,
        SCALER_PATH,
        VIDEO_EXTENSIONS,
        extract_sampled_video_frames,
        extract_visual_feature,
        find_frame_files,
        load_feature_extractor,
        load_json,
        load_projection_model,
        predict_semantic_class,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTION_DESCRIPTIONS_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "action_descriptions.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)


def parse_arguments() -> argparse.Namespace:
    """Read and validate command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict a human action, generate a grounded "
            "natural-language explanation, and optionally save "
            "reports and the exact frames used during inference."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a supported video file or a directory "
            "containing extracted image frames."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of ranked action predictions to display. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help=(
            "Number of evenly spaced frames sampled from a raw "
            "video. Ignored when the input is a frame directory."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of frames processed in one CNN batch. "
            "Default: 16."
        ),
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
        help=(
            "Save a human-readable text report and a structured "
            "JSON metadata file."
        ),
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help=(
            "Save copies of the exact frames used during inference "
            "inside a dedicated visual-evidence folder."
        ),
    )

    parser.add_argument(
        "--save-contact-sheet",
        action="store_true",
        help=(
            "Create one labelled contact-sheet image containing the "
            "exact frames used during inference."
        ),
    )

    parser.add_argument(
        "--contact-columns",
        type=int,
        default=4,
        help=(
            "Number of columns in the contact sheet. Default: 4."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory used for saved reports and visual evidence. "
            "Default: results/predictions."
        ),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional custom base name for saved outputs. "
            "Do not include a file extension."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing outputs with the same name. "
            "Without this option, a numerical suffix is added."
        ),
    )

    arguments = parser.parse_args()

    if arguments.top_k < 1:
        parser.error(
            "--top-k must be at least 1."
        )

    if arguments.num_frames < 1:
        parser.error(
            "--num-frames must be at least 1."
        )

    if arguments.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1."
        )

    if arguments.contact_columns < 1:
        parser.error(
            "--contact-columns must be at least 1."
        )

    if (
        arguments.output_name is not None
        and not arguments.output_name.strip()
    ):
        parser.error(
            "--output-name cannot be empty."
        )

    return arguments


def validate_semantic_metadata(
    class_names: list[str],
    class_embeddings: np.ndarray,
    action_descriptions: dict[str, list[str]],
) -> None:
    """Verify that class names, embeddings, and descriptions match."""

    if not isinstance(
        class_names,
        list,
    ):
        raise TypeError(
            "Class names must be stored as a JSON list."
        )

    if not class_names:
        raise ValueError(
            "Class-name list cannot be empty."
        )

    if class_embeddings.ndim != 2:
        raise ValueError(
            "Class embeddings must be a 2D array."
        )

    if class_embeddings.shape[0] != len(
        class_names
    ):
        raise ValueError(
            "Class-embedding count does not match "
            "the number of class names."
        )

    if not np.isfinite(
        class_embeddings
    ).all():
        raise ValueError(
            "Class embeddings contain NaN or infinite values."
        )

    if not isinstance(
        action_descriptions,
        dict,
    ):
        raise TypeError(
            "Action descriptions must be stored as a JSON object."
        )

    missing_descriptions = [
        class_name
        for class_name in class_names
        if class_name not in action_descriptions
    ]

    if missing_descriptions:
        raise ValueError(
            "Descriptions are missing for classes: "
            f"{missing_descriptions}"
        )


def create_safe_output_stem(
    input_path: Path,
    custom_output_name: str | None = None,
) -> str:
    """
    Create a filesystem-safe output name.

    A custom output name is used when supplied. Otherwise, the input
    filename or frame-folder name becomes the output stem.
    """

    if custom_output_name is not None:
        raw_name = custom_output_name.strip()

        if not raw_name:
            raise ValueError(
                "--output-name cannot be empty."
            )
    else:
        raw_name = (
            input_path.name
            if input_path.is_dir()
            else input_path.stem
        )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        raw_name,
    ).strip("_")

    if not safe_name:
        raise ValueError(
            "Could not create a valid output filename."
        )

    return safe_name


def output_stem_is_available(
    output_directory: Path,
    output_stem: str,
    save_report: bool,
    save_frames: bool,
    save_contact_sheet: bool,
) -> bool:
    """Return True when all requested output paths are unused."""

    requested_paths: list[Path] = []

    if save_report:
        requested_paths.extend(
            [
                output_directory
                / f"{output_stem}_prediction.txt",
                output_directory
                / f"{output_stem}_prediction.json",
            ]
        )

    if save_frames:
        requested_paths.append(
            output_directory
            / f"{output_stem}_frames"
        )

    if save_contact_sheet:
        requested_paths.append(
            output_directory
            / f"{output_stem}_contact_sheet.jpg"
        )

    return all(
        not path.exists()
        for path in requested_paths
    )


def resolve_output_stem(
    input_path: Path,
    output_directory: Path,
    custom_output_name: str | None,
    overwrite: bool,
    save_report: bool,
    save_frames: bool,
    save_contact_sheet: bool,
) -> str:
    """
    Resolve one shared stem for reports and visual-evidence frames.

    Without --overwrite, suffixes such as _2 and _3 are added whenever
    any requested output path already exists.
    """

    base_stem = create_safe_output_stem(
        input_path=input_path,
        custom_output_name=custom_output_name,
    )

    if overwrite:
        return base_stem

    if output_stem_is_available(
        output_directory=output_directory,
        output_stem=base_stem,
        save_report=save_report,
        save_frames=save_frames,
        save_contact_sheet=save_contact_sheet,
    ):
        return base_stem

    suffix = 2

    while True:
        candidate_stem = (
            f"{base_stem}_{suffix}"
        )

        if output_stem_is_available(
            output_directory=output_directory,
            output_stem=candidate_stem,
            save_report=save_report,
            save_frames=save_frames,
            save_contact_sheet=save_contact_sheet,
        ):
            return candidate_stem

        suffix += 1


def build_output_paths(
    output_directory: Path,
    output_stem: str,
) -> tuple[Path, Path, Path, Path]:
    """Build matching report and visual-evidence paths."""

    text_output_path = (
        output_directory
        / f"{output_stem}_prediction.txt"
    )

    json_output_path = (
        output_directory
        / f"{output_stem}_prediction.json"
    )

    frames_output_directory = (
        output_directory
        / f"{output_stem}_frames"
    )

    contact_sheet_output_path = (
        output_directory
        / f"{output_stem}_contact_sheet.jpg"
    )

    return (
        text_output_path,
        json_output_path,
        frames_output_directory,
        contact_sheet_output_path,
    )


def save_inference_frames(
    frame_paths: list[Path],
    frames_output_directory: Path,
    overwrite: bool,
) -> list[Path]:
    """
    Save exact copies of the frames used by the feature extractor.

    Files are renamed in inference order so that the visual evidence is
    easy to inspect and remains independent of temporary source paths.
    """

    if not frame_paths:
        raise ValueError(
            "Cannot save visual evidence because no frames were used."
        )

    if frames_output_directory.exists():
        if not overwrite:
            raise FileExistsError(
                "Frame output directory already exists: "
                f"{frames_output_directory}"
            )

        if frames_output_directory.is_dir():
            shutil.rmtree(
                frames_output_directory
            )
        else:
            frames_output_directory.unlink()

    frames_output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    saved_frame_paths: list[Path] = []

    try:
        for frame_number, source_path in enumerate(
            frame_paths,
            start=1,
        ):
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Inference frame does not exist: {source_path}"
                )

            suffix = (
                source_path.suffix.lower()
                if source_path.suffix
                else ".jpg"
            )

            destination_path = (
                frames_output_directory
                / f"frame_{frame_number:04d}{suffix}"
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

            saved_frame_paths.append(
                destination_path
            )

    except Exception:
        shutil.rmtree(
            frames_output_directory,
            ignore_errors=True,
        )
        raise

    return saved_frame_paths


def create_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    columns: int,
    overwrite: bool,
    tile_width: int = 320,
    tile_height: int = 240,
    caption_height: int = 28,
    spacing: int = 8,
) -> Path:
    """
    Create a labelled grid from the exact frames used for inference.

    Images keep their aspect ratio and are centred inside equal-sized
    tiles. Labels show the inference order from frame 1 onward.
    """

    if not frame_paths:
        raise ValueError(
            "Cannot create a contact sheet because no frames were used."
        )

    if columns < 1:
        raise ValueError(
            "Contact-sheet columns must be at least 1."
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Contact sheet already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = (
        len(frame_paths)
        + columns
        - 1
    ) // columns

    sheet_width = (
        columns * tile_width
        + (columns + 1) * spacing
    )

    tile_total_height = (
        tile_height
        + caption_height
    )

    sheet_height = (
        rows * tile_total_height
        + (rows + 1) * spacing
    )

    contact_sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(
        contact_sheet
    )

    for index, frame_path in enumerate(
        frame_paths
    ):
        if not frame_path.exists():
            raise FileNotFoundError(
                f"Contact-sheet frame does not exist: {frame_path}"
            )

        row = index // columns
        column = index % columns

        tile_x = (
            spacing
            + column * (tile_width + spacing)
        )

        tile_y = (
            spacing
            + row * (tile_total_height + spacing)
        )

        with Image.open(frame_path) as image:
            frame_image = image.convert(
                "RGB"
            )

            fitted_image = ImageOps.contain(
                frame_image,
                (tile_width, tile_height),
            )

        image_canvas = Image.new(
            "RGB",
            (tile_width, tile_height),
            "black",
        )

        paste_x = (
            tile_width
            - fitted_image.width
        ) // 2

        paste_y = (
            tile_height
            - fitted_image.height
        ) // 2

        image_canvas.paste(
            fitted_image,
            (paste_x, paste_y),
        )

        contact_sheet.paste(
            image_canvas,
            (tile_x, tile_y),
        )

        caption = (
            f"Frame {index + 1:02d}"
        )

        caption_y = (
            tile_y
            + tile_height
            + 6
        )

        draw.text(
            (tile_x + 6, caption_y),
            caption,
            fill="black",
        )

    contact_sheet.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return output_path


def build_app_summary(
    explanation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact prediction summary for Streamlit or API use."""

    return {
        "predicted_class": (
            explanation_metadata[
                "predicted_class"
            ]
        ),
        "top_similarity": float(
            explanation_metadata[
                "top_similarity"
            ]
        ),
        "confidence_level": (
            explanation_metadata[
                "confidence_level"
            ]
        ),
        "separation_level": (
            explanation_metadata[
                "separation_level"
            ]
        ),
        "score_gap": float(
            explanation_metadata[
                "score_gap"
            ]
        ),
        "representative_description": (
            explanation_metadata[
                "representative_description"
            ]
        ),
        "explanation": (
            explanation_metadata[
                "explanation"
            ]
        ),
        "reliability_note": (
            explanation_metadata[
                "reliability_note"
            ]
        ),
        "top_predictions": (
            explanation_metadata[
                "top_predictions"
            ]
        ),
    }


def build_complete_report_text(
    input_path: Path,
    input_type: str,
    frames_used: int,
    device: torch.device,
    visual_feature: np.ndarray,
    projected_embedding: np.ndarray,
    explanation_report: str,
    saved_frame_directory: Path | None,
    saved_frame_paths: list[Path],
    contact_sheet_path: Path | None,
) -> str:
    """Build the complete human-readable prediction report."""

    lines = [
        "EXPLAINABLE SEMANTIC VIDEO PREDICTION",
        "=" * 70,
        "",
        "INPUT AND REPRESENTATION",
        "-" * 70,
        f"Input path             : {input_path}",
        f"Input type             : {input_type}",
        f"Frames used            : {frames_used}",
        f"Device                 : {device}",
        (
            "Visual feature shape   : "
            f"{visual_feature.shape}"
        ),
        (
            "Semantic feature shape : "
            f"{projected_embedding.shape}"
        ),
        (
            "Semantic feature norm  : "
            f"{np.linalg.norm(projected_embedding):.6f}"
        ),
    ]

    if (
        saved_frame_directory is not None
        or contact_sheet_path is not None
    ):
        lines.extend(
            [
                "",
                "VISUAL EVIDENCE",
                "-" * 70,
            ]
        )

        if saved_frame_directory is not None:
            lines.extend(
                [
                    (
                        "Saved frame directory  : "
                        f"{saved_frame_directory}"
                    ),
                    (
                        "Saved frame count      : "
                        f"{len(saved_frame_paths)}"
                    ),
                ]
            )

        if contact_sheet_path is not None:
            lines.append(
                "Contact sheet           : "
                f"{contact_sheet_path}"
            )

        lines.append(
            "The visual evidence contains the exact frames used "
            "by the feature extractor."
        )

    lines.extend(
        [
            "",
            explanation_report,
            "",
            "INFERENCE SAFETY",
            "-" * 70,
            (
                "Inference used only the supplied video or frames. "
                "No ground-truth action label was provided."
            ),
        ]
    )

    return "\n".join(
        lines
    )


def build_json_report(
    input_path: Path,
    input_type: str,
    frames_used: int,
    device: torch.device,
    visual_feature: np.ndarray,
    projected_embedding: np.ndarray,
    explanation_metadata: dict[str, Any],
    saved_frame_directory: Path | None,
    saved_frame_paths: list[Path],
    contact_sheet_path: Path | None,
) -> dict[str, Any]:
    """Build a JSON-serializable prediction record."""

    visual_evidence = {
        "frames_saved": (
            saved_frame_directory is not None
        ),
        "saved_frame_count": len(
            saved_frame_paths
        ),
        "saved_frame_directory": (
            str(saved_frame_directory)
            if saved_frame_directory is not None
            else None
        ),
        "saved_frame_paths": [
            str(path)
            for path in saved_frame_paths
        ],
        "contact_sheet_saved": (
            contact_sheet_path is not None
        ),
        "contact_sheet_path": (
            str(contact_sheet_path)
            if contact_sheet_path is not None
            else None
        ),
        "evidence_note": (
            "Saved files are exact copies of the frames used during "
            "visual feature extraction."
            if saved_frame_directory is not None
            else (
                "Visual evidence was not saved for this prediction. "
                "Use --save-frames to preserve the inference frames."
            )
        ),
    }

    return {
        "input_path": str(
            input_path
        ),
        "input_name": input_path.name,
        "input_type": input_type,
        "frames_used": int(
            frames_used
        ),
        "device": str(
            device
        ),
        "visual_feature_shape": list(
            visual_feature.shape
        ),
        "semantic_feature_shape": list(
            projected_embedding.shape
        ),
        "semantic_feature_norm": float(
            np.linalg.norm(
                projected_embedding
            )
        ),
        **explanation_metadata,
        "visual_evidence": visual_evidence,
        "app_summary": build_app_summary(
            explanation_metadata
        ),
    }


def save_prediction_reports(
    text_output_path: Path,
    json_output_path: Path,
    text_report: str,
    json_report: dict[str, Any],
) -> None:
    """Save matching text and JSON reports."""

    text_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_output_path.write_text(
        text_report,
        encoding="utf-8",
    )

    with json_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_report,
            file,
            indent=4,
            ensure_ascii=False,
        )


def print_input_summary(
    input_path: Path,
    input_type: str,
    frames_used: int,
    device: torch.device,
    visual_feature: np.ndarray,
    projected_embedding: np.ndarray,
) -> None:
    """Print model-input and representation information."""

    print(
        f"Input path             : {input_path}"
    )

    print(
        f"Input type             : {input_type}"
    )

    print(
        f"Frames used            : {frames_used}"
    )

    print(
        f"Device                 : {device}"
    )

    print(
        f"Visual feature shape   : {visual_feature.shape}"
    )

    print(
        "Semantic feature shape : "
        f"{projected_embedding.shape}"
    )

    print(
        "Semantic feature norm  : "
        f"{np.linalg.norm(projected_embedding):.6f}"
    )


def run_prediction_from_frames(
    frame_paths: list[Path],
    feature_extractor: torch.nn.Module,
    preprocess: object,
    projection_model: torch.nn.Module,
    scaler: object,
    class_embeddings: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run visual extraction and semantic prediction."""

    visual_feature = extract_visual_feature(
        frame_paths=frame_paths,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
        batch_size=batch_size,
    )

    projected_embedding, similarity_scores = (
        predict_semantic_class(
            visual_feature=visual_feature,
            projection_model=projection_model,
            scaler=scaler,
            class_embeddings=class_embeddings,
            device=device,
        )
    )

    return (
        visual_feature,
        projected_embedding,
        similarity_scores,
    )


def main() -> None:
    arguments = parse_arguments()

    input_path = arguments.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    output_directory = (
        arguments.output_dir.resolve()
    )

    output_stem: str | None = None
    text_output_path: Path | None = None
    json_output_path: Path | None = None
    frames_output_directory: Path | None = None
    contact_sheet_output_path: Path | None = None

    if (
        arguments.save_report
        or arguments.save_frames
        or arguments.save_contact_sheet
    ):
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_stem = resolve_output_stem(
            input_path=input_path,
            output_directory=output_directory,
            custom_output_name=arguments.output_name,
            overwrite=arguments.overwrite,
            save_report=arguments.save_report,
            save_frames=arguments.save_frames,
            save_contact_sheet=arguments.save_contact_sheet,
        )

        (
            text_output_path,
            json_output_path,
            frames_output_directory,
            contact_sheet_output_path,
        ) = build_output_paths(
            output_directory=output_directory,
            output_stem=output_stem,
        )

    print("=" * 70)
    print("EXPLAINABLE SEMANTIC VIDEO PREDICTION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    ).astype(np.float32)

    action_descriptions = load_json(
        ACTION_DESCRIPTIONS_PATH
    )

    validate_semantic_metadata(
        class_names=class_names,
        class_embeddings=class_embeddings,
        action_descriptions=action_descriptions,
    )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Projection scaler not found: {SCALER_PATH}"
        )

    scaler = joblib.load(
        SCALER_PATH
    )

    projection_model = load_projection_model(
        device
    )

    feature_extractor, preprocess = (
        load_feature_extractor(
            device
        )
    )

    input_type: str
    frame_paths: list[Path]
    saved_frame_paths: list[Path] = []

    if input_path.is_dir():
        input_type = "frame folder"

        frame_paths = find_frame_files(
            input_path
        )

        (
            visual_feature,
            projected_embedding,
            similarity_scores,
        ) = run_prediction_from_frames(
            frame_paths=frame_paths,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            projection_model=projection_model,
            scaler=scaler,
            class_embeddings=class_embeddings,
            device=device,
            batch_size=arguments.batch_size,
        )

        if arguments.save_frames:
            if frames_output_directory is None:
                raise RuntimeError(
                    "Frame output directory was not prepared."
                )

            saved_frame_paths = save_inference_frames(
                frame_paths=frame_paths,
                frames_output_directory=frames_output_directory,
                overwrite=arguments.overwrite,
            )

        if arguments.save_contact_sheet:
            if contact_sheet_output_path is None:
                raise RuntimeError(
                    "Contact-sheet output path was not prepared."
                )

            create_contact_sheet(
                frame_paths=frame_paths,
                output_path=contact_sheet_output_path,
                columns=arguments.contact_columns,
                overwrite=arguments.overwrite,
            )

    elif (
        input_path.is_file()
        and input_path.suffix.lower()
        in VIDEO_EXTENSIONS
    ):
        input_type = "video file"

        with tempfile.TemporaryDirectory() as temp_directory:
            temporary_frame_folder = Path(
                temp_directory
            )

            frame_paths = extract_sampled_video_frames(
                video_path=input_path,
                output_folder=temporary_frame_folder,
                number_of_frames=arguments.num_frames,
            )

            (
                visual_feature,
                projected_embedding,
                similarity_scores,
            ) = run_prediction_from_frames(
                frame_paths=frame_paths,
                feature_extractor=feature_extractor,
                preprocess=preprocess,
                projection_model=projection_model,
                scaler=scaler,
                class_embeddings=class_embeddings,
                device=device,
                batch_size=arguments.batch_size,
            )

            if arguments.save_frames:
                if frames_output_directory is None:
                    raise RuntimeError(
                        "Frame output directory was not prepared."
                    )

                # Copy before the temporary directory is deleted.
                saved_frame_paths = save_inference_frames(
                    frame_paths=frame_paths,
                    frames_output_directory=frames_output_directory,
                    overwrite=arguments.overwrite,
                )

            if arguments.save_contact_sheet:
                if contact_sheet_output_path is None:
                    raise RuntimeError(
                        "Contact-sheet output path was not prepared."
                    )

                # Create before temporary sampled frames are deleted.
                create_contact_sheet(
                    frame_paths=frame_paths,
                    output_path=contact_sheet_output_path,
                    columns=arguments.contact_columns,
                    overwrite=arguments.overwrite,
                )

    else:
        raise ValueError(
            "Input must be a supported video file or a directory "
            "containing image frames."
        )

    explanation_metadata = (
        build_explanation_metadata(
            class_names=class_names,
            similarity_scores=similarity_scores,
            action_descriptions=action_descriptions,
            top_k=arguments.top_k,
        )
    )

    print()

    print_input_summary(
        input_path=input_path,
        input_type=input_type,
        frames_used=len(
            frame_paths
        ),
        device=device,
        visual_feature=visual_feature,
        projected_embedding=projected_embedding,
    )

    if arguments.save_frames:
        print(
            "Saved frame count      : "
            f"{len(saved_frame_paths)}"
        )

        print(
            "Saved frame directory  : "
            f"{frames_output_directory}"
        )

    if arguments.save_contact_sheet:
        print(
            "Contact sheet          : "
            f"{contact_sheet_output_path}"
        )

    print()

    explanation_report = (
        format_explanation_report(
            explanation_metadata
        )
    )

    print(
        explanation_report
    )

    complete_text_report = (
        build_complete_report_text(
            input_path=input_path,
            input_type=input_type,
            frames_used=len(
                frame_paths
            ),
            device=device,
            visual_feature=visual_feature,
            projected_embedding=projected_embedding,
            explanation_report=explanation_report,
            saved_frame_directory=(
                frames_output_directory
                if arguments.save_frames
                else None
            ),
            saved_frame_paths=saved_frame_paths,
            contact_sheet_path=(
                contact_sheet_output_path
                if arguments.save_contact_sheet
                else None
            ),
        )
    )

    json_report = build_json_report(
        input_path=input_path,
        input_type=input_type,
        frames_used=len(
            frame_paths
        ),
        device=device,
        visual_feature=visual_feature,
        projected_embedding=projected_embedding,
        explanation_metadata=explanation_metadata,
        saved_frame_directory=(
            frames_output_directory
            if arguments.save_frames
            else None
        ),
        saved_frame_paths=saved_frame_paths,
        contact_sheet_path=(
            contact_sheet_output_path
            if arguments.save_contact_sheet
            else None
        ),
    )

    if arguments.save_report:
        if (
            text_output_path is None
            or json_output_path is None
        ):
            raise RuntimeError(
                "Report output paths were not prepared."
            )

        save_prediction_reports(
            text_output_path=text_output_path,
            json_output_path=json_output_path,
            text_report=complete_text_report,
            json_report=json_report,
        )

        print("\nReports saved:")

        print(
            f"  Text report : {text_output_path}"
        )

        print(
            f"  JSON report : {json_output_path}"
        )

    if (
        arguments.save_frames
        or arguments.save_contact_sheet
    ):
        print("\nVisual evidence saved:")

        if arguments.save_frames:
            print(
                f"  Frame folder : {frames_output_directory}"
            )

            print(
                f"  Frame count  : {len(saved_frame_paths)}"
            )

        if arguments.save_contact_sheet:
            print(
                f"  Contact sheet: {contact_sheet_output_path}"
            )

    print("\n" + "=" * 70)
    print(
        "PASS: EXPLAINABLE VIDEO PREDICTION COMPLETED"
    )
    print("=" * 70)

    print(
        "\nInference used only the supplied video or frames. "
        "No ground-truth action label was provided."
    )


if __name__ == "__main__":
    main()