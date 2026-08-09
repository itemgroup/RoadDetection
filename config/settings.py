"""
全局配置模块
- 模型路径、输入尺寸、类别名称等常量
- 日志初始化
"""

import logging

# --- 模型配置 ---
MODEL_XML = "runs/detect/exp1/weights/best_openvino_model/best.xml"
INPUT_SIZE = (640, 640)

# --- 检测目标类别 ---
CLASS_NAMES = [
    'truck', 'car', 'pedestrian', 'trafficLight-Red', 'trafficLight-RedLeft',
    'biker', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft',
    'trafficLight-YellowLeft', 'trafficLight-Yellow'
]

# --- UI 缩放限制 ---
MAX_ZOOM = 5.0
MIN_ZOOM = 0.1

# --- 默认检测参数 ---
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.45
TIMER_INTERVAL_MS = 30


def setup_logging(log_file: str = "app.log") -> logging.Logger:
    """初始化全局日志配置，返回 root logger。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
