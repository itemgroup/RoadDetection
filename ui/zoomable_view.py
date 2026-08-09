"""
可缩放的 QGraphicsView
按住 Ctrl + 滚轮 实现画面缩放。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView, QApplication


class ZoomableGraphicsView(QGraphicsView):
    """支持 Ctrl+滚轮 缩放的图形视图。"""

    def __init__(self, scene):
        super().__init__(scene)
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # 设置拖拽模式
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def wheelEvent(self, event):
        # 只有在按下Ctrl键时才允许缩放
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
            else:
                self.zoom_factor = max(self.zoom_factor * 0.9, self.min_zoom)

            self.scale(self.zoom_factor, self.zoom_factor)
            self.zoom_factor = 1.0
        else:
            super().wheelEvent(event)
