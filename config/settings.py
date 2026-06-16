import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "animal_detection_model.pth"
MODEL_URL = "https://github.com/Prasad7Paigude/animal-detection-cv/releases/download/v1.0.0/animal_detection_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3
CLASS_LABELS: dict[int, str] = {1: "Carnivorous", 2: "Herbivorous"}
SCORE_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.4

RESIZE_DIM: tuple[int, int] = (640, 640)
INFERENCE_RESIZE_DIM: tuple[int, int] = (640, 640)

TRAIN_BATCH_SIZE: int = 4
VAL_BATCH_SIZE: int = 4
LEARNING_RATE: float = 0.0001
NUM_EPOCHS: int = 10

DATASET_PATHS: dict[str, str] = {
    "train": "",
    "test": "",
    "validate": "",
}
