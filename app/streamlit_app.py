"""Streamlit interface for semantic human action recognition."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "predict_video_explainable.py"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)

SUPPORTED_VIDEO_TYPES = [
    "avi",
    "mp4",
    "mov",
    "mkv",
]


def configure_page() -> None:
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title="Semantic Action Recognition",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the main page title and description."""

    st.title(
        "🎬 Semantic Prompt-Guided Human Action Recognition"
    )

    st.write(
        """
        Upload a video to predict the human action using visual
        features and semantic language embeddings.
        """
    )

    st.info(
        """
        The application combines ResNet18 visual features with
        Sentence-BERT action embeddings through a learned semantic
        projection model.
        """
    )


def render_sidebar() -> tuple[int, int]:
    """Render prediction settings in the sidebar."""

    st.sidebar.header("Prediction Settings")

    top_k = st.sidebar.slider(
        "Number of predictions",
        min_value=1,
        max_value=9,
        value=3,
        step=1,
        help=(
            "Controls how many action candidates are returned."
        ),
    )

    num_frames = st.sidebar.slider(
        "Frames to sample",
        min_value=4,
        max_value=32,
        value=16,
        step=4,
        help=(
            "Controls how many frames are sampled from the video."
        ),
    )

    st.sidebar.divider()

    st.sidebar.subheader("Model Information")

    st.sidebar.write(
        """
        **Visual encoder:** ResNet18  
        **Text encoder:** Sentence-BERT  
        **Fusion method:** Semantic projection  
        **Similarity:** Cosine similarity
        """
    )

    return top_k, num_frames


def validate_project_files() -> None:
    """Stop the application if required project files are missing."""

    if not PREDICTION_SCRIPT.exists():
        st.error(
            "Prediction script was not found."
        )

        st.code(
            str(PREDICTION_SCRIPT)
        )

        st.stop()

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_uploaded_video(
    uploaded_video: Any,
    temporary_directory: Path,
) -> Path:
    """Save the uploaded video inside a temporary directory."""

    original_suffix = (
        Path(uploaded_video.name).suffix.lower()
    )

    if not original_suffix:
        original_suffix = ".avi"

    temporary_video_path = (
        temporary_directory
        / f"uploaded_video{original_suffix}"
    )

    temporary_video_path.write_bytes(
        uploaded_video.getbuffer()
    )

    return temporary_video_path


def create_output_name() -> str:
    """Create a unique prediction output name."""

    unique_id = uuid.uuid4().hex[:10]

    return f"streamlit_prediction_{unique_id}"


def build_prediction_command(
    video_path: Path,
    top_k: int,
    num_frames: int,
    output_name: str,
) -> list[str]:
    """Build the command used to run video prediction."""

    return [
        sys.executable,
        str(PREDICTION_SCRIPT),
        "--input",
        str(video_path),
        "--top-k",
        str(top_k),
        "--num-frames",
        str(num_frames),
        "--save-report",
        "--save-frames",
        "--save-contact-sheet",
        "--contact-columns",
        "4",
        "--output-name",
        output_name,
    ]


def run_prediction(
    video_path: Path,
    top_k: int,
    num_frames: int,
    output_name: str,
) -> subprocess.CompletedProcess[str]:
    """Run the existing explainable prediction script."""

    command = build_prediction_command(
        video_path=video_path,
        top_k=top_k,
        num_frames=num_frames,
        output_name=output_name,
    )

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def find_generated_json_report(
    output_name: str,
) -> Path:
    """Find the JSON report created by the prediction script."""

    exact_report_path = (
        PREDICTIONS_DIR
        / f"{output_name}_prediction.json"
    )

    if exact_report_path.exists():
        return exact_report_path

    matching_reports = sorted(
        PREDICTIONS_DIR.glob(
            f"{output_name}*_prediction.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not matching_reports:
        raise FileNotFoundError(
            "The prediction completed, but no JSON report "
            "was found."
        )

    return matching_reports[0]


def load_json_report(
    report_path: Path,
) -> dict[str, Any]:
    """Load the generated JSON prediction report."""

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(
            "The generated JSON report has an invalid format."
        )

    return report


def render_uploaded_video(
    uploaded_video: Any,
) -> None:
    """Display the selected video and basic file information."""

    st.success(
        f"Video selected: {uploaded_video.name}"
    )

    st.video(uploaded_video)

    file_size_mb = (
        uploaded_video.size
        / (1024 * 1024)
    )

    file_suffix = (
        Path(uploaded_video.name)
        .suffix
        .upper()
        .replace(".", "")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "File size",
            f"{file_size_mb:.2f} MB",
        )

    with col2:
        st.metric(
            "File format",
            file_suffix,
        )


def build_top_predictions_dataframe(
    report: dict[str, Any],
) -> pd.DataFrame:
    """Convert top-prediction metadata into a table."""

    top_predictions = report.get(
        "top_predictions",
        [],
    )

    rows: list[dict[str, Any]] = []

    if not isinstance(
        top_predictions,
        list,
    ):
        return pd.DataFrame(rows)

    for item in top_predictions:
        if not isinstance(
            item,
            dict,
        ):
            continue

        similarity = item.get(
            "similarity"
        )

        if isinstance(
            similarity,
            (int, float),
        ):
            similarity_value = float(
                similarity
            )
        else:
            similarity_value = 0.0

        rows.append(
            {
                "Rank": item.get(
                    "rank",
                    len(rows) + 1,
                ),
                "Action": item.get(
                    "class_name",
                    "Unknown",
                ),
                "Similarity": similarity_value,
            }
        )

    return pd.DataFrame(rows)


def render_top_predictions(
    report: dict[str, Any],
) -> None:
    """Display the ranked top predictions."""

    st.subheader("Top Predictions")

    predictions_df = (
        build_top_predictions_dataframe(
            report
        )
    )

    if predictions_df.empty:
        st.warning(
            "No top prediction ranking was found."
        )
        return

    st.dataframe(
        predictions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d",
            ),
            "Action": st.column_config.TextColumn(
                "Action class",
            ),
            "Similarity": (
                st.column_config.ProgressColumn(
                    "Semantic similarity",
                    min_value=-1.0,
                    max_value=1.0,
                    format="%.4f",
                )
            ),
        },
    )

    st.caption(
        """
        A higher cosine similarity means that the projected semantic
        representation of the video is closer to that action's
        language embedding.
        """
    )


def get_visual_evidence(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Return visual-evidence metadata safely."""

    visual_evidence = report.get(
        "visual_evidence",
        {},
    )

    if not isinstance(
        visual_evidence,
        dict,
    ):
        return {}

    return visual_evidence


def resolve_report_path(
    raw_path: Any,
) -> Path | None:
    """Convert a path stored in the report into a valid Path."""

    if not isinstance(
        raw_path,
        str,
    ):
        return None

    if not raw_path.strip():
        return None

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def render_contact_sheet(
    report: dict[str, Any],
) -> None:
    """Display the contact sheet generated during inference."""

    st.subheader("Visual Evidence")

    visual_evidence = get_visual_evidence(
        report
    )

    contact_sheet_saved = (
        visual_evidence.get(
            "contact_sheet_saved",
            False,
        )
    )

    contact_sheet_path = resolve_report_path(
        visual_evidence.get(
            "contact_sheet_path"
        )
    )

    if not contact_sheet_saved:
        st.warning(
            "The prediction report says that no contact "
            "sheet was saved."
        )
        return

    if contact_sheet_path is None:
        st.warning(
            "The contact-sheet path is missing from "
            "the prediction report."
        )
        return

    if not contact_sheet_path.exists():
        st.warning(
            "The contact-sheet file could not be found."
        )

        st.code(
            str(contact_sheet_path)
        )

        return

    saved_frame_count = visual_evidence.get(
        "saved_frame_count",
        0,
    )

    st.write(
        """
        These are the exact frames sampled from the uploaded video
        and used by the model during inference.
        """
    )

    st.image(
        str(contact_sheet_path),
        caption=(
            f"Inference contact sheet — "
            f"{saved_frame_count} sampled frames"
        ),
        use_container_width=True,
    )

    st.caption(
        f"Contact sheet file: {contact_sheet_path.name}"
    )


def render_inference_frames(
    report: dict[str, Any],
) -> None:
    """Display the individual frames used during inference."""

    visual_evidence = get_visual_evidence(
        report
    )

    frames_saved = visual_evidence.get(
        "frames_saved",
        False,
    )

    raw_frame_paths = visual_evidence.get(
        "saved_frame_paths",
        [],
    )

    if not frames_saved:
        st.warning(
            "The prediction report says that individual "
            "inference frames were not saved."
        )
        return

    if not isinstance(
        raw_frame_paths,
        list,
    ):
        st.warning(
            "The saved-frame metadata has an invalid format."
        )
        return

    valid_frame_paths: list[Path] = []
    missing_frame_paths: list[Path] = []

    for raw_frame_path in raw_frame_paths:
        frame_path = resolve_report_path(
            raw_frame_path
        )

        if frame_path is None:
            continue

        if frame_path.exists():
            valid_frame_paths.append(
                frame_path
            )
        else:
            missing_frame_paths.append(
                frame_path
            )

    if not valid_frame_paths:
        st.warning(
            "No saved inference-frame files could be found."
        )
        return

    with st.expander(
        f"View individual inference frames "
        f"({len(valid_frame_paths)})"
    ):
        st.write(
            """
            These frames were sampled across the uploaded video.
            Their visual features were extracted and aggregated to
            create the video-level representation used for prediction.
            """
        )

        gallery_columns = 4

        for start_index in range(
            0,
            len(valid_frame_paths),
            gallery_columns,
        ):
            row_frame_paths = valid_frame_paths[
                start_index:
                start_index + gallery_columns
            ]

            columns = st.columns(
                gallery_columns
            )

            for column_index, frame_path in enumerate(
                row_frame_paths
            ):
                frame_number = (
                    start_index
                    + column_index
                    + 1
                )

                with columns[column_index]:
                    st.image(
                        str(frame_path),
                        caption=(
                            f"Frame {frame_number:02d}"
                        ),
                        use_container_width=True,
                    )

        st.caption(
            f"Displayed {len(valid_frame_paths)} "
            "saved inference frames."
        )

        if missing_frame_paths:
            st.warning(
                f"{len(missing_frame_paths)} frame files "
                "listed in the report could not be found."
            )

            with st.expander(
                "Show missing frame paths"
            ):
                for missing_path in missing_frame_paths:
                    st.code(
                        str(missing_path)
                    )


def build_prediction_text_summary(
    report: dict[str, Any],
) -> str:
    """Build a readable text summary of one prediction."""

    predicted_class = report.get(
        "predicted_class",
        "Unavailable",
    )

    top_similarity = report.get(
        "top_similarity",
    )

    confidence_level = report.get(
        "confidence_level",
        "Unavailable",
    )

    separation_level = report.get(
        "separation_level",
        "Unavailable",
    )

    score_gap = report.get(
        "score_gap",
    )

    representative_description = report.get(
        "representative_description",
        "Unavailable",
    )

    explanation = report.get(
        "explanation",
        "Unavailable",
    )

    reliability_note = report.get(
        "reliability_note",
        "Unavailable",
    )

    if isinstance(
        top_similarity,
        (int, float),
    ):
        similarity_text = f"{top_similarity:.4f}"
    else:
        similarity_text = "Unavailable"

    if isinstance(
        score_gap,
        (int, float),
    ):
        score_gap_text = f"{score_gap:.4f}"
    else:
        score_gap_text = "Unavailable"

    lines = [
        "SEMANTIC ACTION RECOGNITION REPORT",
        "=" * 40,
        "",
        f"Predicted action      : {predicted_class}",
        f"Semantic similarity   : {similarity_text}",
        f"Confidence level      : {confidence_level}",
        f"Score separation      : {separation_level}",
        f"Top-two score gap     : {score_gap_text}",
        "",
        "REPRESENTATIVE DESCRIPTION",
        "-" * 40,
        str(representative_description),
        "",
        "EXPLANATION",
        "-" * 40,
        str(explanation),
        "",
        "RELIABILITY NOTE",
        "-" * 40,
        str(reliability_note),
        "",
        "TOP PREDICTIONS",
        "-" * 40,
    ]

    top_predictions = report.get(
        "top_predictions",
        [],
    )

    if isinstance(
        top_predictions,
        list,
    ):
        for index, prediction in enumerate(
            top_predictions,
            start=1,
        ):
            if not isinstance(
                prediction,
                dict,
            ):
                continue

            rank = prediction.get(
                "rank",
                index,
            )

            class_name = prediction.get(
                "class_name",
                "Unknown",
            )

            similarity = prediction.get(
                "similarity",
            )

            if isinstance(
                similarity,
                (int, float),
            ):
                prediction_similarity = (
                    f"{similarity:.4f}"
                )
            else:
                prediction_similarity = "Unavailable"

            lines.append(
                f"{rank}. {class_name}: "
                f"{prediction_similarity}"
            )

    visual_evidence = get_visual_evidence(
        report
    )

    lines.extend(
        [
            "",
            "VISUAL EVIDENCE",
            "-" * 40,
            (
                "Frames saved          : "
                f"{visual_evidence.get('frames_saved', False)}"
            ),
            (
                "Saved frame count     : "
                f"{visual_evidence.get('saved_frame_count', 0)}"
            ),
            (
                "Contact sheet saved   : "
                f"{visual_evidence.get('contact_sheet_saved', False)}"
            ),
        ]
    )

    return "\n".join(lines)


def render_report_downloads(
    report: dict[str, Any],
) -> None:
    """Display buttons for downloading prediction reports."""

    st.subheader("Download Prediction Report")

    st.write(
        """
        Save the complete machine-readable report or a shorter
        human-readable summary.
        """
    )

    json_report = json.dumps(
        report,
        indent=4,
        ensure_ascii=False,
    )

    text_report = build_prediction_text_summary(
        report
    )

    predicted_class = str(
        report.get(
            "predicted_class",
            "prediction",
        )
    )

    safe_class_name = (
        predicted_class
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download JSON Report",
            data=json_report,
            file_name=(
                f"{safe_class_name}_prediction_report.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="Download Text Summary",
            data=text_report,
            file_name=(
                f"{safe_class_name}_prediction_summary.txt"
            ),
            mime="text/plain",
            use_container_width=True,
        )


def render_basic_prediction_result(
    report: dict[str, Any],
) -> None:
    """Display the complete prediction result."""

    st.divider()

    predicted_class = report.get(
        "predicted_class",
        "Unavailable",
    )

    st.success(
        f"🎯 Predicted Action: **{predicted_class}**"
    )

    st.subheader("Prediction Details")

    top_similarity = report.get(
        "top_similarity",
    )

    confidence_level = report.get(
        "confidence_level",
        "Unavailable",
    )

    separation_level = report.get(
        "separation_level",
        "Unavailable",
    )

    score_gap = report.get(
        "score_gap",
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted action",
            predicted_class,
        )

    with col2:
        similarity_text = (
            f"{top_similarity:.4f}"
            if isinstance(
                top_similarity,
                (int, float),
            )
            else "Unavailable"
        )

        st.metric(
            "Semantic similarity",
            similarity_text,
        )

    with col3:
        st.metric(
            "Confidence",
            confidence_level,
        )

    with col4:
        score_gap_text = (
            f"{score_gap:.4f}"
            if isinstance(
                score_gap,
                (int, float),
            )
            else "Unavailable"
        )

        st.metric(
            "Top-two score gap",
            score_gap_text,
        )

    st.write(
        f"**Score separation:** {separation_level}"
    )

    representative_description = report.get(
        "representative_description"
    )

    if representative_description:
        st.write(
            "**Representative description:**"
        )

        st.write(
            representative_description
        )

    render_top_predictions(
        report
    )

    explanation = report.get(
        "explanation"
    )

    if explanation:
        st.subheader(
            "Why This Class Was Selected"
        )

        st.write(
            explanation
        )

    reliability_note = report.get(
        "reliability_note"
    )

    if reliability_note:
        st.warning(
            reliability_note
        )

    render_contact_sheet(
        report
    )

    render_inference_frames(
        report
    )

    render_report_downloads(
        report
    )


def render_pipeline_output(
    completed_process: subprocess.CompletedProcess[str],
) -> None:
    """Display prediction-script terminal output for debugging."""

    with st.expander(
        "Prediction Pipeline Output"
    ):
        if completed_process.stdout:
            st.text(
                completed_process.stdout
            )

        if completed_process.stderr:
            st.error(
                completed_process.stderr
            )


def render_video_upload(
    top_k: int,
    num_frames: int,
) -> None:
    """Render video upload and prediction controls."""

    st.subheader("Upload a Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_VIDEO_TYPES,
        help=(
            "Supported formats: AVI, MP4, MOV, and MKV."
        ),
    )

    if uploaded_video is None:
        st.warning(
            "Upload a video to begin action recognition."
        )
        return

    render_uploaded_video(
        uploaded_video
    )

    run_button_clicked = st.button(
        "Run Action Recognition",
        type="primary",
        use_container_width=True,
    )

    if not run_button_clicked:
        return

    output_name = create_output_name()

    try:
        with st.spinner(
            "Running action recognition..."
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                temporary_directory = Path(temp_dir)

                video_path = save_uploaded_video(
                    uploaded_video=uploaded_video,
                    temporary_directory=temporary_directory,
                )

                completed_process = run_prediction(
                    video_path=video_path,
                    top_k=top_k,
                    num_frames=num_frames,
                    output_name=output_name,
                )

        if completed_process.returncode != 0:
            st.error(
                "The prediction pipeline failed."
            )

            render_pipeline_output(
                completed_process
            )

            return

        report_path = find_generated_json_report(
            output_name
        )

        report = load_json_report(
            report_path
        )

        st.success(
            "Action recognition completed successfully."
        )

        render_basic_prediction_result(
            report
        )

        render_pipeline_output(
            completed_process
        )

    except Exception as error:
        st.error(
            "An unexpected error occurred while running "
            "the prediction."
        )

        st.exception(error)


def render_footer() -> None:
    """Render project information at the bottom of the page."""

    st.divider()

    st.caption(
        """
        Developed as part of an Explainable Vision-Language Human
        Action Recognition project.

        Models used: ResNet18, Sentence-BERT, and a Semantic
        Projection Network.

        The similarity scores shown are cosine similarities rather
        than calibrated probabilities.
        """
    )


def main() -> None:
    """Run the Streamlit application."""

    configure_page()
    validate_project_files()
    render_header()

    top_k, num_frames = render_sidebar()

    render_video_upload(
        top_k=top_k,
        num_frames=num_frames,
    )

    render_footer()


if __name__ == "__main__":
    main()