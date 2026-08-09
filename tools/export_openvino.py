"""
PyTorch → OpenVINO 模型导出工具

用法:
    python -m tools.export_openvino --model runs/detect/train/weights/best.pt --output ./openvino_model
    python tools/export_openvino.py --help
"""

import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


# ===================================================================
#  核心导出函数
# ===================================================================

def export_to_openvino(
    model_path: str,
    output_dir: str = 'openvino_model',
    imgsz: int = 640,
    half: bool = True,
    device: str = 'cpu',
    simplify: bool = True,
    task: str = 'detect',
) -> Path:
    """
    将 PyTorch YOLO 模型导出为 OpenVINO IR 格式。

    Args:
        model_path: 训练好的 .pt 模型路径
        output_dir: OpenVINO 模型输出目录
        imgsz: 输入图像尺寸
        half: 是否使用 FP16
        device: 导出设备
        simplify: 是否 ONNX 简化
        task: 任务类型 (detect / segment / classify / pose)

    Returns:
        导出的 OpenVINO 模型目录
    """
    from ultralytics import YOLO

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)

    logger.info("Exporting to OpenVINO...")
    model.export(
        format='openvino',
        imgsz=imgsz,
        half=half,
        device=device,
        simplify=simplify,
        task=task,
    )

    out_dir = Path(output_dir)
    logger.info("Done. OpenVINO model saved to: %s", out_dir)

    # 验证导出结果
    xml_files = list(out_dir.rglob('*.xml'))
    if xml_files:
        logger.info("Found model: %s", xml_files[0])
    return out_dir


# ===================================================================
#  CLI 入口
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='YOLO PyTorch → OpenVINO 模型导出'
    )
    parser.add_argument('--model', required=True, help='训练好的 .pt 模型路径')
    parser.add_argument('--output', default='openvino_model', help='输出目录')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸 (默认: 640)')
    parser.add_argument('--half', action='store_true', default=True, help='FP16 精度')
    parser.add_argument('--fp32', action='store_true', help='使用 FP32 (覆盖 --half)')
    parser.add_argument('--device', default='cpu', help='导出设备 (默认: cpu)')
    parser.add_argument('--no-simplify', action='store_true', help='跳过 ONNX 简化')
    parser.add_argument('--task', default='detect',
                        choices=['detect', 'segment', 'classify', 'pose'],
                        help='任务类型 (默认: detect)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    use_half = args.half and not args.fp32
    use_simplify = not args.no_simplify

    export_to_openvino(
        model_path=args.model,
        output_dir=args.output,
        imgsz=args.imgsz,
        half=use_half,
        device=args.device,
        simplify=use_simplify,
        task=args.task,
    )


if __name__ == '__main__':
    main()
