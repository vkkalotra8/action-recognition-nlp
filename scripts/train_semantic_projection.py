from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_ROOT = PROJECT_ROOT / "data" / "features"
TEXT_ROOT = PROJECT_ROOT / "data" / "text_embeddings"
MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "results" / "reports"

CLASS_EMBEDDINGS_PATH = (
    TEXT_ROOT / "class_embeddings.npy"
)

CLASS_NAMES_PATH = (
    TEXT_ROOT / "class_names.json"
)

MODEL_OUTPUT_PATH = (
    MODELS_ROOT / "semantic_projection_model.pth"
)

SCALER_OUTPUT_PATH = (
    MODELS_ROOT / "semantic_projection_scaler.joblib"
)

METADATA_OUTPUT_PATH = (
    MODELS_ROOT / "semantic_projection_metadata.json"
)

REPORT_OUTPUT_PATH = (
    REPORTS_ROOT / "semantic_projection_training_report.txt"
)

RANDOM_SEED = 42
INPUT_DIMENSION = 512
OUTPUT_DIMENSION = 384
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30


class SemanticProjectionModel(nn.Module):
    """Project visual features into the Sentence-BERT semantic space."""

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


def set_random_seeds(seed: int) -> None:
    """Make training as reproducible as possible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(file_path: Path):
    """Load a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split(
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load visual features and labels for one split."""

    features_path = (
        VISUAL_ROOT / f"{split_name}_features.npy"
    )

    labels_path = (
        VISUAL_ROOT / f"{split_name}_labels.npy"
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}"
        )

    features = np.load(
        features_path
    ).astype(np.float32)

    labels = np.load(
        labels_path,
        allow_pickle=True,
    )

    return features, labels


def build_target_embeddings(
    labels: np.ndarray,
    class_names: list[str],
    class_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Build semantic training targets from class labels.

    These embeddings are targets only. They are not concatenated
    with the visual input features.
    """

    label_to_embedding = {
        class_name: class_embeddings[index]
        for index, class_name in enumerate(class_names)
    }

    target_embeddings = np.stack(
        [
            label_to_embedding[str(label)]
            for label in labels
        ],
        axis=0,
    )

    return target_embeddings.astype(np.float32)


def cosine_distance_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Minimize one minus cosine similarity.

    A perfect match produces a loss close to zero.
    """

    cosine_scores = nn.functional.cosine_similarity(
        predictions,
        targets,
        dim=1,
    )

    return (
        1.0 - cosine_scores
    ).mean()


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Calculate average loss without updating model weights."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)

            loss = cosine_distance_loss(
                predictions,
                targets,
            )

            batch_size = features.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Validation data loader contains no samples."
        )

    return total_loss / total_samples


def calculate_nearest_class_accuracy(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    class_embeddings: np.ndarray,
    device: torch.device,
) -> float:
    """
    Project visual features and classify them using cosine similarity
    against every class text embedding.
    """

    model.eval()

    feature_tensor = torch.from_numpy(
        features
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
        projected_features = model(
            feature_tensor
        )

        similarity_scores = (
            projected_features
            @ class_embedding_tensor.T
        )

        predicted_indices = similarity_scores.argmax(
            dim=1
        ).cpu().numpy()

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    true_indices = np.array(
        [
            class_to_index[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )

    return float(
        np.mean(predicted_indices == true_indices)
    )


def main() -> None:
    print("=" * 70)
    print("TRAINING LEAKAGE-SAFE VISUAL-TO-SEMANTIC PROJECTION")
    print("=" * 70)

    set_random_seeds(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    class_embeddings = np.load(
        CLASS_EMBEDDINGS_PATH
    ).astype(np.float32)

    train_features, train_labels = load_split(
        "train"
    )

    val_features, val_labels = load_split(
        "val"
    )

    if train_features.ndim != 2:
        raise ValueError(
            "Training features must be a 2D array."
        )

    if val_features.ndim != 2:
        raise ValueError(
            "Validation features must be a 2D array."
        )

    if train_labels.ndim != 1:
        raise ValueError(
            "Training labels must be a 1D array."
        )

    if val_labels.ndim != 1:
        raise ValueError(
            "Validation labels must be a 1D array."
        )

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError(
            "Training feature count does not match label count."
        )

    if val_features.shape[0] != val_labels.shape[0]:
        raise ValueError(
            "Validation feature count does not match label count."
        )

    if train_features.shape[1] != INPUT_DIMENSION:
        raise ValueError(
            f"Expected train dimension {INPUT_DIMENSION}, "
            f"received {train_features.shape[1]}."
        )

    if val_features.shape[1] != INPUT_DIMENSION:
        raise ValueError(
            f"Expected validation dimension {INPUT_DIMENSION}, "
            f"received {val_features.shape[1]}."
        )

    if class_embeddings.shape != (
        len(class_names),
        OUTPUT_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected class embedding shape: "
            f"{class_embeddings.shape}"
        )

    if not np.isfinite(train_features).all():
        raise ValueError(
            "Training features contain invalid values."
        )

    if not np.isfinite(val_features).all():
        raise ValueError(
            "Validation features contain invalid values."
        )

    if not np.isfinite(class_embeddings).all():
        raise ValueError(
            "Class embeddings contain invalid values."
        )

    dataset_classes = set(
        np.concatenate(
            [
                train_labels,
                val_labels,
            ]
        ).tolist()
    )

    text_classes = set(class_names)

    if dataset_classes != text_classes:
        raise ValueError(
            "Dataset labels and text-embedding classes do not match."
        )

    scaler = StandardScaler()

    train_features_scaled = scaler.fit_transform(
        train_features
    ).astype(np.float32)

    val_features_scaled = scaler.transform(
        val_features
    ).astype(np.float32)

    train_targets = build_target_embeddings(
        train_labels,
        class_names,
        class_embeddings,
    )

    val_targets = build_target_embeddings(
        val_labels,
        class_names,
        class_embeddings,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_features_scaled),
        torch.from_numpy(train_targets),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_features_scaled),
        torch.from_numpy(val_targets),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SemanticProjectionModel().to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_history: list[
        dict[str, float | int]
    ] = []

    best_model_state: (
        dict[str, torch.Tensor] | None
    ) = None

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        total_training_loss = 0.0
        total_training_samples = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(
                features
            )

            loss = cosine_distance_loss(
                predictions,
                targets,
            )

            loss.backward()
            optimizer.step()

            batch_size = features.shape[0]

            total_training_loss += (
                loss.item() * batch_size
            )

            total_training_samples += batch_size

        if total_training_samples == 0:
            raise ValueError(
                "Training data loader contains no samples."
            )

        training_loss = (
            total_training_loss
            / total_training_samples
        )

        validation_loss = evaluate_loss(
            model,
            val_loader,
            device,
        )

        validation_accuracy = (
            calculate_nearest_class_accuracy(
                model=model,
                features=val_features_scaled,
                labels=val_labels,
                class_names=class_names,
                class_embeddings=class_embeddings,
                device=device,
            )
        )

        training_history.append(
            {
                "epoch": epoch,
                "training_loss": float(
                    training_loss
                ),
                "validation_loss": float(
                    validation_loss
                ),
                "validation_accuracy": float(
                    validation_accuracy
                ),
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
                f"train_loss={training_loss:.6f} | "
                f"val_loss={validation_loss:.6f} | "
                f"val_accuracy={validation_accuracy:.4f}"
            )

        accuracy_improved = (
            validation_accuracy
            > best_validation_accuracy + 1e-6
        )

        accuracy_tied_loss_improved = (
            abs(
                validation_accuracy
                - best_validation_accuracy
            ) <= 1e-6
            and validation_loss
            < best_validation_loss - 1e-6
        )

        if (
            accuracy_improved
            or accuracy_tied_loss_improved
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= (
            EARLY_STOPPING_PATIENCE
        ):
            print(
                f"\nEarly stopping at epoch {epoch}."
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "No best model state was recorded."
        )

    model.load_state_dict(
        best_model_state
    )

    final_validation_accuracy = (
        calculate_nearest_class_accuracy(
            model=model,
            features=val_features_scaled,
            labels=val_labels,
            class_names=class_names,
            class_embeddings=class_embeddings,
            device=device,
        )
    )

    final_validation_loss = evaluate_loss(
        model,
        val_loader,
        device,
    )

    MODELS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "input_dimension": INPUT_DIMENSION,
            "output_dimension": OUTPUT_DIMENSION,
            "class_names": class_names,
            "checkpoint_selection": (
                "highest_validation_accuracy_then_"
                "lowest_validation_loss"
            ),
            "best_epoch": best_epoch,
            "best_validation_accuracy": (
                best_validation_accuracy
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
        },
        MODEL_OUTPUT_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_OUTPUT_PATH,
    )

    metadata = {
        "model_type": (
            "linear_visual_to_semantic_projection"
        ),
        "leakage_safe": True,
        "checkpoint_selection": (
            "highest_validation_accuracy_then_"
            "lowest_validation_loss"
        ),
        "input_dimension": INPUT_DIMENSION,
        "output_dimension": OUTPUT_DIMENSION,
        "loss": (
            "mean_one_minus_cosine_similarity"
        ),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "random_seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(
            best_validation_accuracy
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "final_validation_accuracy": float(
            final_validation_accuracy
        ),
        "final_validation_loss": float(
            final_validation_loss
        ),
        "device": str(device),
        "training_history": (
            training_history
        ),
    }

    with METADATA_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    report_lines = [
        (
            "LEAKAGE-SAFE VISUAL-TO-SEMANTIC "
            "PROJECTION TRAINING"
        ),
        "=" * 70,
        "",
        f"Device: {device}",
        (
            f"Training samples: "
            f"{train_features.shape[0]}"
        ),
        (
            f"Validation samples: "
            f"{val_features.shape[0]}"
        ),
        (
            f"Input dimension: "
            f"{INPUT_DIMENSION}"
        ),
        (
            f"Output dimension: "
            f"{OUTPUT_DIMENSION}"
        ),
        (
            "Checkpoint selection: highest validation "
            "accuracy, then lowest validation loss"
        ),
        f"Best epoch: {best_epoch}",
        (
            "Best validation accuracy: "
            f"{best_validation_accuracy:.4f}"
        ),
        (
            "Best validation loss: "
            f"{best_validation_loss:.6f}"
        ),
        (
            "Reloaded-model validation accuracy: "
            f"{final_validation_accuracy:.4f}"
        ),
        (
            "Reloaded-model validation loss: "
            f"{final_validation_loss:.6f}"
        ),
        "",
        (
            "This model receives only visual features "
            "during inference."
        ),
        (
            "Ground-truth text embeddings are used as "
            "training targets, not as input features."
        ),
    ]

    REPORT_OUTPUT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best epoch                  : "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy    : "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Best validation loss        : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Nearest-class val accuracy  : "
        f"{final_validation_accuracy:.4f}"
    )

    print(
        f"Reloaded validation loss    : "
        f"{final_validation_loss:.6f}"
    )

    print("\nFiles saved:")

    print(
        f"  Model    : {MODEL_OUTPUT_PATH}"
    )

    print(
        f"  Scaler   : {SCALER_OUTPUT_PATH}"
    )

    print(
        f"  Metadata : {METADATA_OUTPUT_PATH}"
    )

    print(
        f"  Report   : {REPORT_OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "PASS: LEAKAGE-SAFE PROJECTION MODEL TRAINED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()