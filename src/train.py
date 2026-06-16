import logging

import numpy as np
import torch
import torchvision
import torchvision.models.detection as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

from config.settings import (
    NUM_CLASSES,
    DEVICE,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODEL_PATH,
    DATASET_PATHS,
)
from src.dataset import AnimalDataset, eval_transform, train_transform

logger = logging.getLogger(__name__)


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    return tuple(zip(*batch))


def build_model() -> torch.nn.Module:
    model = models.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, NUM_CLASSES
        )
    )
    model.to(DEVICE)
    return model


def train() -> None:
    logger.info("Starting training pipeline")

    train_dataset = AnimalDataset(
        DATASET_PATHS["train"], transforms=train_transform
    )
    val_dataset = AnimalDataset(
        DATASET_PATHS["validate"], transforms=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [
                {k: v.to(DEVICE) for k, v in t.items()} for t in targets
            ]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(
            "Epoch %d/%d, Loss: %.4f", epoch + 1, NUM_EPOCHS, epoch_loss
        )

    torch.save(model.state_dict(), str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)


def evaluate() -> dict[str, float]:
    logger.info("Starting evaluation")

    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    val_dataset = AnimalDataset(
        DATASET_PATHS["validate"], transforms=eval_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    true_labels: list[int] = []
    pred_labels: list[int] = []

    score_threshold = 0.5

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_classes = (
                    output["labels"][output["scores"] > score_threshold]
                    .cpu()
                    .numpy()
                )
                true_classes = targets[i]["labels"].cpu().numpy()

                min_length = min(len(true_classes), len(pred_classes))
                true_labels.extend(true_classes[:min_length])
                pred_labels.extend(pred_classes[:min_length])

    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)

    accuracy = accuracy_score(true_arr, pred_arr) * 100
    precision = precision_score(true_arr, pred_arr, average="weighted") * 100
    recall = recall_score(true_arr, pred_arr, average="weighted") * 100
    conf_matrix = confusion_matrix(true_arr, pred_arr)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix,
    }

    logger.info(
        "Model Accuracy: %.2f%%\nPrecision: %.2f%%\nRecall: %.2f%%\nConfusion Matrix:\n%s",
        accuracy,
        precision,
        recall,
        conf_matrix,
    )

    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()
