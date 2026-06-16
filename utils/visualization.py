import logging
import numpy as np
import cv2
from PIL import Image
import PyQt5.QtGui as qtg

from config.settings import CLASS_LABELS

logger = logging.getLogger(__name__)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        label = labels[i]

        color = (0, 0, 255) if label == 1 else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{CLASS_LABELS[int(label)]}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image


def pil_to_qt(image: Image.Image) -> qtg.QImage:
    image = image.convert("RGB")
    data = np.array(image)
    height, width, channel = data.shape
    bytes_per_line = 3 * width
    return qtg.QImage(data.data, width, height, bytes_per_line, qtg.QImage.Format_RGB888)
