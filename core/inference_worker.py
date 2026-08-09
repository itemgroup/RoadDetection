"""
推理工作线程
在子线程中完成 预处理 → 推理 → 后处理，避免阻塞 UI。
"""

import logging
import cv2
import numpy as np
from PyQt5.QtCore import QThread

from .signals import WorkerSignals
from .detector import OVDetector
from config import INPUT_SIZE, CLASS_NAMES

logger = logging.getLogger(__name__)


class InferenceWorkerThread(QThread):
    """独立的推理线程，处理单帧图像的目标检测。"""

    def __init__(self, frame, detector, conf_thresh, nms_thresh):
        super().__init__()
        self.frame = frame.copy()
        self.detector = detector
        self.conf_threshold = conf_thresh
        self.nms_threshold = nms_thresh
        self.signals = WorkerSignals()

    def run(self):
        try:
            logger.info("Starting inference processing...")
            input_data, scale, (pad_h, pad_w) = self.preprocess(self.frame)
            output = self.detector.infer(input_data)
            detections = self.postprocess(output, scale, pad_h, pad_w, self.frame.shape[:2])
            self.signals.finished.emit(self.frame, detections)
            logger.info("Inference completed successfully")
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            self.signals.error.emit(f"Inference error: {str(e)}")

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE[0]/h, INPUT_SIZE[1]/w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        input_blob = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), 114, dtype=np.uint8)
        pad_h = (INPUT_SIZE[0] - new_h) // 2
        pad_w = (INPUT_SIZE[1] - new_w) // 2
        input_blob[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        input_blob = input_blob.transpose((2, 0, 1))
        input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32) / 255.0
        return input_blob, scale, (pad_h, pad_w)

    def postprocess(self, outputs, scale, pad_h, pad_w, orig_shape):
        outputs = np.transpose(outputs, (0, 2, 1))[0]
        x_center = outputs[:, 0]
        y_center = outputs[:, 1]
        w = outputs[:, 2]
        h = outputs[:, 3]
        x1 = ((x_center - w/2 - pad_w) / scale).clip(0, orig_shape[1])
        y1 = ((y_center - h/2 - pad_h) / scale).clip(0, orig_shape[0])
        x2 = ((x_center + w/2 - pad_w) / scale).clip(0, orig_shape[1])
        y2 = ((y_center + h/2 - pad_h) / scale).clip(0, orig_shape[0])
        class_scores = outputs[:, 4:4+len(CLASS_NAMES)]
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores > self.conf_threshold
        if not np.any(valid_mask):
            return []
        boxes = np.stack([x1, y1, x2, y2], axis=1)[valid_mask].astype(int)
        scores = max_scores[valid_mask]
        class_ids = np.argmax(class_scores[valid_mask], axis=1)
        try:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                     self.conf_threshold, self.nms_threshold).flatten()
        except:
            indices = []
        return [np.concatenate((box, [score], [class_id]))
                for box, score, class_id in zip(boxes[indices], scores[indices], class_ids[indices])]
