import logging
import os
import urllib.error

import cv2
import numpy as np
import torch
import torch.hub
import torchvision
import torchvision.ops as ops

from config.settings import (
    NUM_CLASSES,
    SCORE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    MODEL_PATH,
    MODEL_URL,
)

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


def download_model_if_missing() -> None:
    if MODEL_PATH.exists():
        return

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    torch.hub.download_url_to_file(MODEL_URL, str(MODEL_PATH))


def load_model() -> torch.nn.Module:
    try:
        download_model_if_missing()

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, NUM_CLASSES
            )
        )

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    except FileNotFoundError:
        logger.error("Model file not found at %s", MODEL_PATH)
        raise ModelLoadError(
            f"Model file not found at {MODEL_PATH}. "
            f"Place 'animal_detection_model.pth' in the models/ directory."
        )
    except (RuntimeError, urllib.error.URLError, OSError) as e:
        logger.error("Failed to prepare or load model: %s", e)
        raise ModelLoadError(f"Failed to load model weights from {MODEL_PATH}: {e}")
    except Exception as e:
        logger.error("Failed to load model state dict: %s", e)
        raise ModelLoadError(
            f"Failed to load model weights from {MODEL_PATH}: {e}"
        )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    logger.info("Model loaded from %s on device %s", MODEL_PATH, DEVICE)
    return model


def process_frame(
    image: np.ndarray,
    model: torch.nn.Module,
) -> tuple[np.ndarray, int]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = (
        torch.tensor(image_rgb / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        output = model(image_tensor)[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    keep = scores > SCORE_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) == 0:
        return image, 0

    keep_indices = (
        ops.nms(torch.tensor(boxes), torch.tensor(scores), IOU_THRESHOLD)
        .numpy()
    )
    boxes, scores, labels = (
        boxes[keep_indices],
        scores[keep_indices],
        labels[keep_indices],
    )

    carnivore_count = int(sum(1 for label in labels if label == 1))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        label = labels[i]

        color = (0, 0, 255) if label == 1 else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{'Carnivorous' if label == 1 else 'Herbivorous'}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image, carnivore_count
