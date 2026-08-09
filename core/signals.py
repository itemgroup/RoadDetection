"""
Qt 信号定义模块
统一管理跨线程通信的信号，避免在业务代码中混杂 UI 信号声明。
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    """推理工作线程信号：完成或出错时向外通知。"""
    finished = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)


class ProcessingSignals(QObject):
    """图像处理线程信号：处理完成后向外传递结果帧。"""
    result_ready = pyqtSignal(np.ndarray)
