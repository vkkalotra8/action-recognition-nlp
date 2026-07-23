from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
MODELS_ROOT = PROJECT_ROOT / "models"

CLASS_EMBEDDINGS_PATH = (
    TEXT_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    TEXT_ROOT / "class_names.json"
)

PROJECTION_MODEL_PATH = (
    MODELS_ROOT / "semantic_projection_model.pth"
)

SCALER_PATH = (
    MODELS_ROOT / "semantic_projection_scaler.joblib"
)

INPUT_DIMENSION = 512
OUTPUT_DIMENSION = 384

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}


class SemanticProjectionModel(nn.Module):
    """Project visual features into the Sentence-BERT space."""

    def __init__(
        self,
        input_dimension: int = INPUT_DIMENSION,
        output_dimension: int = OUTPUT_DIMENSION,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(
            input_dimension,
            output_dimension,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(features)

        return nn.functional.normalize(
            projected,
            p=2,
            dim=1,
        )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict an action class using the leakage-safe "
            "visual-to-semantic projection model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a video file or a folder containing "
            "extracted image frames."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to display. Default: 3.",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help=(
            "Number of evenly spaced frames to sample from a video. "
            "Ignored when the input is already a frame folder."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of frames processed together. Default: 16.",
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

    return arguments


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_feature_extractor(
    device: torch.device,
) -> tuple[nn.Module, object]:
    """Load pretrained ResNet18 without its classifier."""

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    )

    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()

    return model, preprocess


def find_frame_files(
    frame_folder: Path,
) -> list[Path]:
    """Return supported image files in one frame folder."""

    frame_paths = sorted(
        path
        for path in frame_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not frame_paths:
        raise ValueError(
            f"No supported image frames found in: {frame_folder}"
        )

    return frame_paths


def select_evenly_spaced_indices(
    total_frames: int,
    number_to_select: int,
) -> np.ndarray:
    """Choose evenly spaced frame indices."""

    if total_frames <= 0:
        raise ValueError(
            "The video contains no readable frames."
        )

    number_to_select = min(
        number_to_select,
        total_frames,
    )

    return np.linspace(
        0,
        total_frames - 1,
        number_to_select,
        dtype=np.int64,
    )


def extract_sampled_video_frames(
    video_path: Path,
    output_folder: Path,
    number_of_frames: int,
) -> list[Path]:
    """Extract evenly spaced RGB frames from one video."""

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        capture.release()

        raise ValueError(
            f"Video contains no readable frames: {video_path}"
        )

    selected_indices = (
        select_evenly_spaced_indices(
            total_frames=total_frames,
            number_to_select=number_of_frames,
        )
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_paths: list[Path] = []

    for output_index, frame_index in enumerate(
        selected_indices,
        start=1,
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        success, frame = capture.read()

        if not success:
            continue

        frame_path = (
            output_folder
            / f"frame_{output_index:04d}.jpg"
        )

        write_success = cv2.imwrite(
            str(frame_path),
            frame,
        )

        if write_success:
            frame_paths.append(
                frame_path
            )

    capture.release()

    if not frame_paths:
        raise ValueError(
            f"No frames could be extracted from: {video_path}"
        )

    return frame_paths


def load_frame_batch(
    frame_paths: list[Path],
    preprocess: object,
) -> torch.Tensor:
    """Load and preprocess a group of image frames."""

    processed_frames = []

    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb_image = image.convert("RGB")
            processed_frame = preprocess(
                rgb_image
            )

        processed_frames.append(
            processed_frame
        )

    return torch.stack(
        processed_frames
    )


def extract_visual_feature(
    frame_paths: list[Path],
    feature_extractor: nn.Module,
    preprocess: object,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Create one mean-pooled 512-dimensional video feature."""

    frame_feature_batches = []

    for start_index in range(
        0,
        len(frame_paths),
        batch_size,
    ):
        batch_paths = frame_paths[
            start_index:
            start_index + batch_size
        ]

        input_batch = load_frame_batch(
            frame_paths=batch_paths,
            preprocess=preprocess,
        ).to(device)

        with torch.inference_mode():
            batch_features = feature_extractor(
                input_batch
            )

        frame_feature_batches.append(
            batch_features.cpu()
        )

    all_frame_features = torch.cat(
        frame_feature_batches,
        dim=0,
    )

    visual_feature = all_frame_features.mean(
        dim=0
    )

    visual_feature_array = (
        visual_feature.numpy().astype(
            np.float32
        )
    )

    if visual_feature_array.shape != (
        INPUT_DIMENSION,
    ):
        raise ValueError(
            "Unexpected visual feature shape: "
            f"{visual_feature_array.shape}"
        )

    return visual_feature_array


def load_projection_model(
    device: torch.device,
) -> SemanticProjectionModel:
    """Load the trained visual-to-semantic model."""

    if not PROJECTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Projection model not found: "
            f"{PROJECTION_MODEL_PATH}"
        )

    checkpoint = torch.load(
        PROJECTION_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = SemanticProjectionModel(
        input_dimension=int(
            checkpoint["input_dimension"]
        ),
        output_dimension=int(
            checkpoint["output_dimension"]
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict_semantic_class(
    visual_feature: np.ndarray,
    projection_model: SemanticProjectionModel,
    scaler: object,
    class_embeddings: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the visual feature and calculate class similarities."""

    scaled_feature = scaler.transform(
        visual_feature.reshape(1, -1)
    ).astype(np.float32)

    visual_tensor = torch.from_numpy(
        scaled_feature
    ).to(device)

    class_embedding_tensor = torch.from_numpy(
        class_embeddings
    ).to(device)

    class_embedding_tensor = nn.functional.normalize(
        class_embedding_tensor,
        p=2,
        dim=1,
    )

    with torch.no_grad():
        projected_embedding = projection_model(
            visual_tensor
        )

        similarity_scores = (
            projected_embedding
            @ class_embedding_tensor.T
        )

    return (
        projected_embedding.cpu().numpy()[0],
        similarity_scores.cpu().numpy()[0],
    )


def print_predictions(
    class_names: list[str],
    similarity_scores: np.ndarray,
    top_k: int,
) -> None:
    """Print ranked class predictions."""

    number_to_show = min(
        top_k,
        len(class_names),
    )

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    print(
        f"\nTop {number_to_show} predictions:"
    )

    for rank, class_index in enumerate(
        ranked_indices[:number_to_show],
        start=1,
    ):
        print(
            f"  {rank}. "
            f"{class_names[class_index]:<15} "
            f"similarity={similarity_scores[class_index]:.4f}"
        )

    top_index = int(
        ranked_indices[0]
    )

    print(
        f"\nPredicted action: "
        f"{class_names[top_index]}"
    )


def main() -> None:
    arguments = parse_arguments()

    input_path = arguments.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    print("=" * 70)
    print("LEAKAGE-SAFE SEMANTIC VIDEO PREDICTION")
    print("=" * 70)

    print(
        f"Input path       : {input_path}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device           : {device}"
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    ).astype(np.float32)

    if class_embeddings.shape != (
        len(class_names),
        OUTPUT_DIMENSION,
    ):
        raise ValueError(
            "Unexpected class-embedding shape: "
            f"{class_embeddings.shape}"
        )

    if not np.isfinite(
        class_embeddings
    ).all():
        raise ValueError(
            "Class embeddings contain invalid values."
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

    if input_path.is_dir():
        frame_paths = find_frame_files(
            input_path
        )

        print(
            "Input type       : frame folder"
        )

        print(
            f"Frames available : {len(frame_paths)}"
        )

        visual_feature = extract_visual_feature(
            frame_paths=frame_paths,
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
            batch_size=arguments.batch_size,
        )

    elif (
        input_path.is_file()
        and input_path.suffix.lower()
        in VIDEO_EXTENSIONS
    ):
        print(
            "Input type       : video file"
        )

        with tempfile.TemporaryDirectory() as temp_directory:
            temporary_frame_folder = Path(
                temp_directory
            )

            frame_paths = extract_sampled_video_frames(
                video_path=input_path,
                output_folder=temporary_frame_folder,
                number_of_frames=arguments.num_frames,
            )

            print(
                f"Frames sampled   : {len(frame_paths)}"
            )

            visual_feature = extract_visual_feature(
                frame_paths=frame_paths,
                feature_extractor=feature_extractor,
                preprocess=preprocess,
                device=device,
                batch_size=arguments.batch_size,
            )

    else:
        raise ValueError(
            "Input must be a supported video file or "
            "a folder containing image frames."
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

    print(
        f"Visual feature    : {visual_feature.shape}"
    )

    print(
        f"Semantic feature  : {projected_embedding.shape}"
    )

    print(
        "Semantic norm     : "
        f"{np.linalg.norm(projected_embedding):.6f}"
    )

    print_predictions(
        class_names=class_names,
        similarity_scores=similarity_scores,
        top_k=arguments.top_k,
    )

    print("\n" + "=" * 70)
    print("PASS: SEMANTIC VIDEO PREDICTION COMPLETED")
    print("=" * 70)

    print(
        "\nInference used only the input video or frames. "
        "No true action label was supplied."
    )


if __name__ == "__main__":
    main()