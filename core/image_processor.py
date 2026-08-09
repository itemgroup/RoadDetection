"""
图像处理模块
封装所有图像处理效果（灰度、模糊、均衡化等）以及对应的后台线程。
"""

import cv2
import numpy as np
from PyQt5.QtCore import QThread

from .signals import ProcessingSignals


class ProcessingThread(QThread):
    """在子线程中对帧应用指定的图像处理效果。"""

    def __init__(self, frame, index):
        super().__init__()
        self.frame = frame.copy() if frame is not None else None
        self.index = index
        self.signals = ProcessingSignals()

    def run(self):
        try:
            if self.frame is None or not np.any(self.frame):
                raise ValueError("输入帧为空或无效")

            processed_frame = self.process_frame()
            self.signals.result_ready.emit(processed_frame)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"处理线程错误: {str(e)}")
            # 发送一个默认的空帧，避免程序崩溃
            self.signals.result_ready.emit(np.zeros((100,100,3), dtype=np.uint8))

    def process_frame(self):
        if self.frame is None or not np.any(self.frame):
            return np.zeros((100,100,3), dtype=np.uint8)

        processed_frame = self.frame.copy()

        if self.index == 0:  # 视频 - 原始帧
            pass
        elif self.index == 1:  # 灰度化
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        elif self.index == 2:  # 平滑处理
            processed_frame = cv2.GaussianBlur(processed_frame, (15, 15), 0)
        elif self.index == 3:  # 均衡化
            ycrcb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            processed_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        elif self.index == 4:  # 图像梯度
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            processed_frame = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        elif self.index == 5:  # 阈值处理
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            processed_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        elif self.index == 6:  # 边缘检测
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.index == 7:  # 轮廓检测
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(processed_frame, contours, -1, (0, 255, 0), 2)
        elif self.index == 8:  # 直线检测
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif self.index == 9:  # 亮度调节
            alpha = 1.5  # 控制对比度
            beta = 50    # 控制亮度
            processed_frame = cv2.convertScaleAbs(processed_frame, alpha=alpha, beta=beta)
        elif self.index == 10:  # 伽马校正
            gamma = 0.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
            processed_frame = cv2.LUT(processed_frame, table)

        return processed_frame


# 为了兼容 main_window 中的旧引用，保留以下常量
IMAGE_PROCESSORS = [
    ("原始视频",  None),
    ("灰度化",    None),
    ("平滑处理",  None),
    ("均衡化",    None),
    ("图像梯度",  None),
    ("阈值处理",  None),
    ("边缘检测",  None),
    ("轮廓检测",  None),
    ("直线检测",  None),
    ("亮度调节",  None),
    ("伽马校正",  None),
]

PROCESSOR_NAMES = [name for name, _ in IMAGE_PROCESSORS]
