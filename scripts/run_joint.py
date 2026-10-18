"""Script to test and run JointTrainer with MultiPlaneKneeModel."""
import torch
from src.models.multi_plane import MultiPlaneKneeModel
from src.training.joint_trainer import JointTrainer


def main() -> None:
    print("Initializing MultiPlaneKneeModel...")
    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_classes=12,
    )

    print("Initializing JointTrainer...")
    trainer = JointTrainer(
        model=model,
        checkpoint_dir="checkpoints/joint_experiment",
    )
    print(f"✓ JointTrainer successfully initialized on device: {trainer.device}")


if __name__ == "__main__":
    main()
