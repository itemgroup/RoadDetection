"""
YOLO 模型训练 / 验证 / 推理工具

用法:
    # 训练
    python -m tools.train --mode train --model yolov8n.yaml --data data.yaml --epochs 200

    # 验证
    python -m tools.train --mode val --model best.pt --data data.yaml

    # 推理
    python -m tools.train --mode predict --model best.pt --source images/test

    # 恢复训练
    python -m tools.train --mode resume --model last.pt
"""

import logging
import argparse

logger = logging.getLogger(__name__)


# ===================================================================
#  训练
# ===================================================================

def run_train(
    model_path: str,
    data_yaml: str,
    *,
    epochs: int = 200,
    batch: int = 8,
    workers: int = 16,
    imgsz: int = 640,
    optimizer: str = 'SGD',
    project: str | None = None,
    name: str | None = None,
    pretrained: str | None = None,
) -> None:
    from ultralytics import YOLO

    logger.info("Creating model from: %s", model_path)
    model = YOLO(model_path)
    if pretrained:
        logger.info("Loading pretrained weights: %s", pretrained)
        model.load(pretrained)

    logger.info("Training started | epochs=%d | batch=%d", epochs, batch)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        workers=workers,
        imgsz=imgsz,
        optimizer=optimizer,
        project=project,
        name=name,
    )
    logger.info("Training completed")


# ===================================================================
#  验证
# ===================================================================

def run_validate(
    model_path: str,
    data_yaml: str,
    *,
    split: str = 'test',
    project: str | None = None,
    name: str | None = None,
) -> None:
    from ultralytics import YOLO

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)
    logger.info("Validating on split: %s", split)
    model.val(data=data_yaml, split=split, project=project, name=name)
    logger.info("Validation completed")


# ===================================================================
#  推理
# ===================================================================

def run_predict(
    model_path: str,
    source: str,
    *,
    save: bool = True,
    project: str | None = None,
    name: str | None = None,
) -> None:
    from ultralytics import YOLO

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)
    logger.info("Running inference on: %s", source)
    model.predict(source=source, save=save, project=project, name=name)
    logger.info("Inference completed")


# ===================================================================
#  恢复训练
# ===================================================================

def run_resume(model_path: str) -> None:
    from ultralytics import YOLO

    logger.info("Resuming training from: %s", model_path)
    model = YOLO(model_path)
    model.train(resume=True)


# ===================================================================
#  CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO 训练 / 验证 / 推理工具')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'val', 'predict', 'resume'],
                        help='运行模式')
    parser.add_argument('--model', required=True, help='模型路径或 yaml 配置')
    parser.add_argument('--data', help='data.yaml 路径 (train/val 模式必需)')
    parser.add_argument('--source', help='推理源路径 (predict 模式必需)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数 (默认: 200)')
    parser.add_argument('--batch', type=int, default=8, help='批次大小 (默认: 8)')
    parser.add_argument('--workers', type=int, default=16, help='数据加载线程 (默认: 16)')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸 (默认: 640)')
    parser.add_argument('--optimizer', default='SGD', help='优化器 (默认: SGD)')
    parser.add_argument('--project', help='输出项目目录')
    parser.add_argument('--name', help='实验名称')
    parser.add_argument('--pretrained', help='预训练权重路径 (train 模式可选)')
    parser.add_argument('--split', default='test', help='验证集 split (默认: test)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.mode == 'train':
        if not args.data:
            parser.error('--data is required for train mode')
        run_train(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            imgsz=args.imgsz,
            optimizer=args.optimizer,
            project=args.project,
            name=args.name,
            pretrained=args.pretrained,
        )
    elif args.mode == 'val':
        if not args.data:
            parser.error('--data is required for val mode')
        run_validate(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            project=args.project,
            name=args.name,
        )
    elif args.mode == 'predict':
        if not args.source:
            parser.error('--source is required for predict mode')
        run_predict(
            model_path=args.model,
            source=args.source,
            project=args.project,
            name=args.name,
        )
    elif args.mode == 'resume':
        run_resume(model_path=args.model)


if __name__ == '__main__':
    main()
