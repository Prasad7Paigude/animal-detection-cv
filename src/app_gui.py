import logging

import cv2
import torch
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qt
from PIL import Image
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout

from src.inference import load_model, process_frame, ModelLoadError
from utils.visualization import pil_to_qt

logger = logging.getLogger(__name__)


class VideoProcessor(qtc.QObject):
    frame_processed = qtc.pyqtSignal(object, int)
    finished = qtc.pyqtSignal()

    def __init__(self, video_path: str, model: torch.nn.Module) -> None:
        super().__init__()
        self.video_path = video_path
        self.model = model
        self._is_running = True

    @qtc.pyqtSlot()
    def process(self) -> None:
        try:
            cap = cv2.VideoCapture(self.video_path)
        except Exception as e:
            logger.error("Failed to open video %s: %s", self.video_path, e)
            self.finished.emit()
            return

        try:
            while self._is_running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame, carnivore_count = process_frame(
                        frame, self.model
                    )
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    img_qt = pil_to_qt(img_pil)
                    self.frame_processed.emit(img_qt, carnivore_count)
                    qtc.QThread.msleep(25)
                except Exception as e:
                    logger.error(
                        "Error processing video frame: %s", e
                    )
                    break
        finally:
            cap.release()
            self.finished.emit()

    def stop(self) -> None:
        self._is_running = False


class AnimalDetectionApp(qt.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.cap = None
        self.thread = None
        self.worker = None
        self.init_ui()
        self._load_model_safe()

    def init_ui(self) -> None:
        self.setWindowTitle("Animal Detection GUI")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label_title = QLabel("Animal Detection System")
        self.label_title.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.label_title)

        self.btn_image = QPushButton("Select Image")
        self.btn_image.clicked.connect(self.select_image)
        layout.addWidget(self.btn_image)

        self.btn_video = QPushButton("Select Video")
        self.btn_video.clicked.connect(self.select_video)
        layout.addWidget(self.btn_video)

        self.image_label = QLabel()
        self.image_label.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.count_label = QLabel("Carnivorous Count: 0")
        self.count_label.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.count_label)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def _load_model_safe(self) -> None:
        try:
            self.model = load_model()
        except ModelLoadError as e:
            logger.critical(str(e))
            qt.QMessageBox.critical(
                self,
                "Model Load Error",
                f"Could not load the detection model.\n\n{e}\n\n"
                f"The application will now close.",
            )
            qt.QApplication.quit()

    def select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File")
        if not file_path:
            return

        image = cv2.imread(file_path)
        processed_image, carnivore_count = process_frame(image, self.model)

        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(processed_image)
        img_qt = pil_to_qt(img_pil)

        self.image_label.setPixmap(
            qtg.QPixmap.fromImage(img_qt).scaled(
                400, 400, qtc.Qt.KeepAspectRatio
            )
        )
        self.count_label.setText(f"Carnivorous Count: {carnivore_count}")

    def select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if not file_path:
            return

        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()

        self.thread = qtc.QThread()
        self.worker = VideoProcessor(file_path, self.model)
        self.worker.moveToThread(self.thread)
        self.worker.frame_processed.connect(self._update_video_frame)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.process)
        self.thread.start()

    @qtc.pyqtSlot(object, int)
    def _update_video_frame(self, img_qt: qtg.QImage, count: int) -> None:
        self.image_label.setPixmap(
            qtg.QPixmap.fromImage(img_qt).scaled(
                500, 500, qtc.Qt.KeepAspectRatio
            )
        )
        self.count_label.setText(f"Carnivorous Count: {count}")

    def stop_video(self) -> None:
        if self.worker:
            self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.image_label.clear()
        self.count_label.setText("Carnivorous Count: 0")
        self.stop_button.setEnabled(False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = qt.QApplication([])
    window = AnimalDetectionApp()
    window.show()
    app.exec_()
