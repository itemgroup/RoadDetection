"""
OpenVINO 检测器封装
负责模型加载和同步推理，不涉及任何 Qt 或 UI 逻辑。
"""

import logging
import numpy as np
from openvino.runtime import Core

logger = logging.getLogger(__name__)


class OVDetector:
    """基于 OpenVINO 的目标检测推理引擎。"""

    def __init__(self):
        self.ie = Core()
        self.model = None
        self.compiled_model = None

    def load_model(self, model_path):
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = self.ie.read_model(model_path)
            self.compiled_model = self.ie.compile_model(self.model, "CPU")
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            return False

    def infer(self, input_data):
        return self.compiled_model([input_data])[self.compiled_model.output(0)]
