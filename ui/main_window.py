"""
主窗口
负责 UI 布局、用户交互和线程调度。
"""

import sys
import time
import logging

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QMutex, QFileInfo
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QFileDialog, QMessageBox, QSlider,
    QSizePolicy, QSplitter, QTabBar, QLabel, QLineEdit,
    QInputDialog, QListWidget, QGraphicsScene,
)

from config import (
    MODEL_XML, CLASS_NAMES, setup_logging,
    DEFAULT_CONF_THRESHOLD, DEFAULT_NMS_THRESHOLD, TIMER_INTERVAL_MS,
)
from core import (
    OVDetector, InferenceWorkerThread, ProcessingThread,
    IMAGE_PROCESSORS, PROCESSOR_NAMES,
)
from .zoomable_view import ZoomableGraphicsView

logger = setup_logging()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = OVDetector()
        self.model_paths = {}
        self.cap = None
        self.is_processing = False
        self.video_file = None
        self.available_cameras = []
        self.last_frame = None
        self.last_processed_frame = None
        self.detections = []
        self.processed_detections = []
        self.input_source = None
        self.selected_button = None
        self.frame_lock = QMutex()
        self.active_processing = None
        self.active_inference_thread = None

        # 用于帧率和延迟计算
        self.fps_start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.latency = 0

        self.init_ui()
        self.load_default_model()
        self.detect_cameras()

    def init_ui(self):
        self.setWindowTitle("基于openvino的道路障碍识别系统")
        self.setGeometry(100, 100, 1280, 720)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部标题栏
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #2c3e50; color: white; height: 40px;")
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(10, 5, 10, 5)

        self.title_label = QLabel("基于openvino的道路障碍识别系统")
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))

        title_bar_layout.addWidget(self.title_label)
        title_bar.setLayout(title_bar_layout)
        title_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 顶部导航栏
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #34495e; color: white; height: 40px;")
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(10, 5, 10, 5)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("视频")
        self.tab_bar.addTab("灰度化")
        self.tab_bar.addTab("平滑处理")
        self.tab_bar.addTab("均衡化")
        self.tab_bar.addTab("图像梯度")
        self.tab_bar.addTab("阈值处理")
        self.tab_bar.addTab("边缘检测")
        self.tab_bar.addTab("轮廓检测")
        self.tab_bar.addTab("直线检测")
        self.tab_bar.addTab("亮度调节")
        self.tab_bar.addTab("伽马校正")
        self.tab_bar.setStyleSheet("QTabBar::tab { padding: 5px 15px; margin: 0px 5px; }")

        top_bar_layout.addWidget(self.tab_bar)
        top_bar.setLayout(top_bar_layout)
        top_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 主要布局区域
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        # 左侧设置面板
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #34495e; color: white;")
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(10, 10, 10, 10)
        left_panel_layout.setSpacing(15)

        # 模型管理
        model_group = QWidget()
        model_group.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("模型管理"))
        model_layout.addWidget(QLabel("选择已加载的模型"))

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("background-color: white; color: black; padding: 5px;")
        model_layout.addWidget(self.model_combo)

        load_model_btn = QPushButton("加载模型")
        load_model_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px 20px;")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)

        model_group.setLayout(model_layout)

        # 输入源
        input_group = QWidget()
        input_group.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("输入源"))
        input_layout.addWidget(QLabel("输入类型:"))

        self.input_combo = QComboBox()
        self.input_combo.addItems(["摄像头", "图片", "视频"])
        self.input_combo.setStyleSheet("background-color: white; color: black; padding: 5px;")
        input_layout.addWidget(self.input_combo)

        self.file_btn = QPushButton("选择文件")
        self.file_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px 20px;")
        self.file_btn.clicked.connect(self.select_file)
        input_layout.addWidget(self.file_btn)

        input_layout.addWidget(QLabel("摄像头选择:"))

        self.camera_combo = QComboBox()
        self.camera_combo.setStyleSheet("background-color: white; color: black; padding: 5px;")
        input_layout.addWidget(self.camera_combo)

        refresh_cam_btn = QPushButton("刷新摄像头")
        refresh_cam_btn.setStyleSheet("background-color: #3498db; color: white; padding: 5px 20px;")
        refresh_cam_btn.clicked.connect(self.detect_cameras)
        input_layout.addWidget(refresh_cam_btn)

        input_group.setLayout(input_layout)

        # 检测参数
        param_group = QWidget()
        param_group.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        param_layout = QVBoxLayout()
        param_layout.addWidget(QLabel("检测参数"))

        self.conf_slider = self.create_slider("置信度阈值", 0, 100, 25)
        self.nms_slider = self.create_slider("NMS阈值", 0, 100, 45)

        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.nms_slider)

        param_group.setLayout(param_layout)

        # 启动检测按钮
        self.control_btn = QPushButton("启动检测")
        self.control_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px 30px;")
        self.control_btn.clicked.connect(self.toggle_processing)

        left_panel_layout.addWidget(model_group)
        left_panel_layout.addWidget(input_group)
        left_panel_layout.addWidget(param_group)
        left_panel_layout.addWidget(self.control_btn)
        left_panel_layout.addStretch(1)

        left_panel.setLayout(left_panel_layout)

        # 右侧显示面板
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #ecf0f1;")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # 分割右侧面板为上下两部分
        upper_right_layout = QHBoxLayout()
        upper_right_splitter = QSplitter(Qt.Horizontal)

        # 原始画面
        original_group = QWidget()
        original_group.setStyleSheet("background-color: white; border-radius: 5px;")
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("原始画面"))

        self.original_scene = QGraphicsScene()
        self.original_view = ZoomableGraphicsView(self.original_scene)
        self.original_view.setStyleSheet("background-color: #333;")

        original_layout.addWidget(self.original_view)
        original_group.setLayout(original_layout)

        # 检测结果
        result_group = QWidget()
        result_group.setStyleSheet("background-color: white; border-radius: 5px;")
        result_layout = QVBoxLayout()
        result_layout.addWidget(QLabel("检测结果"))

        self.result_scene = QGraphicsScene()
        self.result_view = ZoomableGraphicsView(self.result_scene)
        self.result_view.setStyleSheet("background-color: #333;")

        result_layout.addWidget(self.result_view)
        result_group.setLayout(result_layout)

        upper_right_splitter.addWidget(original_group)
        upper_right_splitter.addWidget(result_group)
        upper_right_splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.5)])
        upper_right_splitter.setStretchFactor(0, 1)
        upper_right_splitter.setStretchFactor(1, 1)

        upper_right_layout.addWidget(upper_right_splitter)
        upper_right_layout.setContentsMargins(0, 0, 0, 0)

        # 检测结果统计
        stats_group = QWidget()
        stats_group.setStyleSheet("background-color: #ecf0f1; border-radius: 5px;")
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel("检测结果统计"))

        stats_container = QWidget()
        stats_container.setStyleSheet("background-color: white; border-radius: 5px; padding: 5px;")
        stats_container_layout = QHBoxLayout()

        self.result_list = QListWidget()
        self.result_list.setStyleSheet("background-color: white; color: black;")
        self.result_list.setMaximumHeight(100)

        # 添加FPS和延迟显示
        self.fps_label = QLabel("帧率: 0 FPS")
        self.latency_label = QLabel("延迟: 0 ms")

        stats_container_layout.addWidget(self.result_list)
        stats_container_layout.addWidget(self.fps_label)
        stats_container_layout.addWidget(self.latency_label)
        stats_container.setLayout(stats_container_layout)

        stats_layout.addWidget(stats_container)
        stats_group.setLayout(stats_layout)

        right_layout.addLayout(upper_right_layout)
        right_layout.addWidget(stats_group)
        right_layout.setStretch(0, 3)
        right_layout.setStretch(1, 1)

        right_panel.setLayout(right_layout)

        # 使用QSplitter实现左右布局
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])

        main_layout.addWidget(title_bar)
        main_layout.addWidget(top_bar)
        main_layout.addWidget(main_splitter)

        self.input_combo.currentTextChanged.connect(self.handle_input_change)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.setInterval(30)

        self.tab_bar.currentChanged.connect(self.handle_tab_change)

    def create_slider(self, title, min_val, max_val, default):
        container = QWidget()
        layout = QVBoxLayout()
        slider_layout = QHBoxLayout()

        label = QLabel(title)
        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(200)
        value = QLabel(f"{default/100:.2f}")

        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v: value.setText(f"{v/100:.2f}"))

        slider_layout.addWidget(label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value)

        container.setLayout(slider_layout)
        return container

    def load_default_model(self):
        try:
            if self.detector.load_model(MODEL_XML):
                self.model_combo.addItem("默认模型")
        except Exception as e:
            self.show_error("模型加载", f"默认模型加载失败: {str(e)}")

    def detect_cameras(self):
        self.available_cameras.clear()
        self.camera_combo.clear()

        for index in range(10):
            try:
                backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened() and cap.read()[0]:
                    self.available_cameras.append(str(index))
                    cap.release()
            except Exception as e:
                logger.warning(f"摄像头 {index} 检测失败: {str(e)}")

        if self.available_cameras:
            self.camera_combo.addItems(self.available_cameras)
        else:
            self.camera_combo.setEditable(True)
            self.camera_combo.addItem("手动输入")
            self.camera_combo.setToolTip("可输入摄像头设备号或视频文件路径")

    def handle_input_change(self, text):
        self.camera_combo.setEnabled(text == "摄像头")
        self.file_btn.setEnabled(text != "摄像头")

    def load_model(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.xml)")
            if file and self.detector.load_model(file):
                model_name = QFileInfo(file).fileName()
                self.model_combo.addItem(model_name)
                self.show_message("成功", f"模型加载成功：{model_name}")
        except Exception as e:
            self.show_error("加载失败", str(e))

    def toggle_processing(self):
        if self.is_processing:
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.is_processing = True

        try:
            source_type = self.input_combo.currentText()
            if source_type == "摄像头":
                cam_input = self.camera_combo.currentText()
                if cam_input == "手动输入":
                    cam_input, ok = QInputDialog.getText(
                        self, "手动输入", "请输入摄像头设备号或视频路径:"
                    )
                    if not ok:
                        return
                try:
                    cam_index = int(cam_input)
                    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2
                    self.cap = cv2.VideoCapture(cam_index, backend)
                except ValueError:
                    self.cap = cv2.VideoCapture(cam_input)
                if not self.cap.isOpened():
                    raise RuntimeError("无法打开输入源")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            elif source_type == "视频" and self.video_file:
                self.cap = cv2.VideoCapture(self.video_file)
                if not self.cap.isOpened():
                    raise RuntimeError("无法打开视频文件")
            elif source_type == "图片" and self.video_file:
                frame = cv2.imread(self.video_file)
                if frame is not None:
                    self.show_frame(frame, self.original_scene)
                    self.run_inference(frame)
                    self.stop_processing()
                return
            self.control_btn.setText("停止检测")
            self.timer.start()
        except Exception as e:
            self.show_error("启动失败", str(e))
            self.stop_processing()

    def stop_processing(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.active_processing and self.active_processing.isRunning():
            self.active_processing.quit()
            self.active_processing.wait()
        if self.active_inference_thread and self.active_inference_thread.isRunning():
            self.active_inference_thread.quit()
            self.active_inference_thread.wait()
        self.is_processing = False
        self.control_btn.setText("启动检测")

    def process_frame(self):
        try:
            source_type = self.input_combo.currentText()
            if source_type in ["视频", "摄像头"] and self.cap and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()
                if ret:
                    # 显示原始帧
                    self.show_frame(frame, self.original_scene)

                    # 获取当前选中的处理选项
                    current_tab = self.tab_bar.currentIndex()

                    # 处理当前帧
                    if self.active_processing and self.active_processing.isRunning():
                        self.active_processing.quit()
                        self.active_processing.wait()
                    self.active_processing = ProcessingThread(frame, current_tab)
                    self.active_processing.signals.result_ready.connect(self.run_inference_on_processed_frame)
                    self.active_processing.start()
                else:
                    if source_type == "视频":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if ret:
                            self.show_frame(frame, self.original_scene)
                            self.run_inference(frame)
                    else:
                        self.stop_processing()

                # 计算帧率和延迟
                self.update_fps_and_latency(start_time)
        except Exception as e:
            logger.error(f"帧处理错误: {str(e)}")
            self.stop_processing()

    def run_inference_on_processed_frame(self, processed_frame):
        try:
            if processed_frame is None:
                logger.warning("无有效处理结果")
                return
            if processed_frame.size == 0:
                logger.warning("收到空帧")
                return

            if len(processed_frame.shape) == 2:
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame = processed_frame

            # 显示处理后的帧
            self.show_frame(display_frame, self.original_scene)

            # 对处理后的帧进行目标检测
            if self.active_inference_thread and self.active_inference_thread.isRunning():
                self.active_inference_thread.quit()
                self.active_inference_thread.wait()

            conf_thresh = self.conf_slider.findChild(QSlider).value() / 100
            nms_thresh = self.nms_slider.findChild(QSlider).value() / 100

            self.active_inference_thread = InferenceWorkerThread(display_frame, self.detector, conf_thresh, nms_thresh)
            self.active_inference_thread.signals.finished.connect(self.handle_processed_frame_results)
            self.active_inference_thread.signals.error.connect(self.show_error)
            self.active_inference_thread.start()

        except Exception as e:
            logger.error(f"显示处理结果错误: {str(e)}")

    def handle_processed_frame_results(self, frame, detections):
        self.last_processed_frame = frame.copy()
        self.processed_detections = detections

        self.update_processed_detection_stats()

        # 在检测结果视图中显示处理后的帧和检测结果
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            score = det[4]
            cid = int(det[5])
            class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "未知"

            color = (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.show_frame(result_frame, self.result_scene)

    def run_inference(self, frame):
        if self.active_inference_thread and self.active_inference_thread.isRunning():
            self.active_inference_thread.quit()
            self.active_inference_thread.wait()

        conf_thresh = self.conf_slider.findChild(QSlider).value() / 100
        nms_thresh = self.nms_slider.findChild(QSlider).value() / 100

        self.active_inference_thread = InferenceWorkerThread(frame, self.detector, conf_thresh, nms_thresh)
        self.active_inference_thread.signals.finished.connect(self.handle_results)
        self.active_inference_thread.signals.error.connect(self.show_error)
        self.active_inference_thread.start()

    def handle_results(self, frame, detections):
        self.last_frame = frame.copy()
        self.detections = detections

        self.update_detection_stats()

        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            score = det[4]
            cid = int(det[5])
            class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "未知"

            color = (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.show_frame(result_frame, self.result_scene)

    def update_detection_stats(self):
        count = {}
        if hasattr(self, 'detections'):
            for det in self.detections:
                cid = int(det[5])
                class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "未知"
                count[class_name] = count.get(class_name, 0) + 1

        self.result_list.clear()
        total = sum(count.values()) if hasattr(self, 'detections') else 0
        self.result_list.addItem(f"总计: {total} 个物体")

        for class_name, num in count.items():
            self.result_list.addItem(f"{class_name}: {num} 个")

    def update_processed_detection_stats(self):
        count = {}
        if hasattr(self, 'processed_detections'):
            for det in self.processed_detections:
                cid = int(det[5])
                class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "未知"
                count[class_name] = count.get(class_name, 0) + 1

        self.result_list.clear()
        total = sum(count.values()) if hasattr(self, 'processed_detections') else 0
        self.result_list.addItem(f"总计: {total} 个物体")

        for class_name, num in count.items():
            self.result_list.addItem(f"{class_name}: {num} 个")

    def show_frame(self, frame, scene):
        try:
            if frame is None:
                logger.warning("尝试显示空帧")
                return
            if frame.size == 0:
                logger.warning("收到空帧")
                return
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

        # 调整图像大小以适应显示区域
            h, w = frame.shape[:2]
        # 计算缩放比例
            scene_width = self.original_view.viewport().width() if scene == self.original_scene else self.result_view.viewport().width()
            scene_height = self.original_view.viewport().height() if scene == self.original_scene else self.result_view.viewport().height()
            scale = min(scene_width / w, scene_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))

            rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

        # 清除场景并添加新的图像
            scene.clear()
            scene.addPixmap(pixmap)

        # 设置显示区域的大小与图片匹配
            scene.setSceneRect(scene.itemsBoundingRect())

        except Exception as e:
            logger.error(f"显示错误: {str(e)}")

    def select_file(self):
        try:
            source_type = self.input_combo.currentText()
            if source_type == "图片":
                file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
                if file:
                    self.video_file = file
                    frame = cv2.imread(self.video_file)
                    if frame is not None:
                        self.show_frame(frame, self.original_scene)
                        # 直接对图片进行目标检测
                        self.run_inference(frame)
            elif source_type == "视频":
                file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
                if file:
                    self.video_file = file
                    cap = cv2.VideoCapture(file)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            self.show_frame(frame, self.original_scene)
                        cap.release()
        except Exception as e:
            self.show_error("文件错误", str(e))

    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")

    def show_message(self, title, message):
        QMessageBox.information(self, title, message)
        logger.info(f"{title}: {message}")

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

    def update_fps_and_latency(self, start_time):
        current_time = time.time()
        elapsed = current_time - start_time
        self.latency = int(elapsed * 1000)  # 转换为毫秒

        self.frame_count += 1
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = current_time

        self.fps_label.setText(f"帧率: {self.fps} FPS")
        self.latency_label.setText(f"延迟: {self.latency} ms")

    def handle_tab_change(self, index):
        try:
            source_type = self.input_combo.currentText()
            if source_type in ["视频", "摄像头", "图片"]:
                if self.active_processing and self.active_processing.isRunning():
                    self.active_processing.quit()
                    self.active_processing.wait()

                if self.last_frame is not None and source_type in ["图片", "视频", "摄像头"]:
                    self.active_processing = ProcessingThread(self.last_frame, index)
                    self.active_processing.signals.result_ready.connect(self.run_inference_on_processed_frame)
                    self.active_processing.start()
        except Exception as e:
            logger.error(f"处理标签切换时发生错误: {str(e)}", exc_info=True)
            self.show_error("标签切换错误", f"处理标签切换时发生错误: {str(e)}")
