import platform
import sys


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print_section("1. Python Information")

    print(f"Python version : {sys.version}")
    print(f"Python path    : {sys.executable}")
    print(f"Operating system: {platform.platform()}")

    print_section("2. PyTorch Check")

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version     : {torch.version.cuda}")
            print(f"GPU count        : {torch.cuda.device_count()}")
            print(f"GPU name         : {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("No CUDA GPU detected.")
            print("The project will currently use the CPU.")
            device = torch.device("cpu")

        print(f"Selected device  : {device}")

    except ImportError:
        print("PyTorch is not installed.")
        print("Install PyTorch before continuing.")
        return

    print_section("3. Torchvision Check")

    try:
        import torchvision

        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision is not installed.")
        return

    print_section("4. ResNet18 Loading Test")

    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()

        print("ResNet18 loaded successfully.")
        print("Final classification layer was removed.")
        print("Expected feature size: 512")

    except Exception as error:
        print("ResNet18 could not be loaded.")
        print(f"Error: {error}")
        return

    print_section("5. Dummy Feature Extraction Test")

    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"Input shape  : {tuple(dummy_input.shape)}")
        print(f"Output shape : {tuple(output.shape)}")

        if output.shape == (1, 512):
            print("Environment check passed successfully.")
        else:
            print("Unexpected ResNet18 output shape.")

    except Exception as error:
        print("Dummy feature extraction failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()