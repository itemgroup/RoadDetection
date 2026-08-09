"""
摄像头工具
"""

import cv2
import logging

logger = logging.getLogger(__name__)


def list_cameras(max_index: int = 10) -> list[int]:
    """扫描可用摄像头，返回可用设备号列表。"""
    import sys
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2
    cameras = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened() and cap.read()[0]:
            cameras.append(idx)
        cap.release()
    return cameras


def test_camera(camera_id: int = 0) -> None:
    """打开指定摄像头并显示一帧测试。"""
    import sys
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_V4L2
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", camera_id)
        return

    logger.info("Camera %d opened successfully", camera_id)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Test', frame)
        logger.info("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.error("Cannot read frame from camera %d", camera_id)
    cap.release()
