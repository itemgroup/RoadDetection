"""
BDD100K → YOLO 格式数据集转换工具

用法:
    python -m tools.dataset_converter --images E:/AI/bdd100k/images --labels E:/AI/bdd100k/labels --output ./bdd_yolo
    python tools/dataset_converter.py --help
"""

import json
import logging
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  默认配置
# ---------------------------------------------------------------------------
DEFAULT_CLASS_MAP = {
    'bus': 0,
    'traffic light': 1,
    'traffic sign': 2,
    'person': 3,
    'bike': 4,
    'truck': 5,
    'motor': 6,
    'car': 7,
    'rider': 8,
}


# ---------------------------------------------------------------------------
class BDD2YOLOConverter:
    """将 BDD100K 标注转换为 YOLO txt 格式。"""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        class_map: dict | None = None,
        splits: tuple[str, ...] = ('train', 'val', 'test'),
        image_ext: str = '.jpg',
    ):
        self._images_dir = images_dir
        self._labels_dir = labels_dir
        self._output_dir = output_dir
        self._class_map = class_map or DEFAULT_CLASS_MAP
        self._splits = splits
        self._image_ext = image_ext

    # ------------------------------------------------------------------
    def run(self) -> None:
        for split in self._splits:
            self._process_split(split)
        self._generate_yaml()
        logger.info("Conversion completed")

    # ------------------------------------------------------------------
    def _process_split(self, split: str) -> None:
        split_dir = self._labels_dir / split
        if not split_dir.exists():
            logger.warning("Split dir not found, skipped: %s", split_dir)
            return

        json_files = list(split_dir.glob('*.json'))
        if not json_files:
            logger.warning("No JSON files in %s", split_dir)
            return

        logger.info("Processing %s (%d files)...", split, len(json_files))
        for json_path in tqdm(json_files):
            self._convert_one(json_path, split)

    # ------------------------------------------------------------------
    def _convert_one(self, json_path: Path, split: str) -> None:
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, IOError) as exc:
            logger.error("Failed to read %s: %s", json_path, exc)
            return

        img_name = data.get('name', json_path.stem) + self._image_ext
        img_path = self._images_dir / split / img_name
        if not img_path.exists():
            logger.warning("Missing image: %s", img_path)
            return

        # 获取原图尺寸
        with Image.open(img_path) as img:
            iw, ih = img.size

        # 写 YOLO 标注
        label_path = self._output_dir / 'labels' / split / f"{data['name']}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for frame in data.get('frames', []):
            for obj in frame.get('objects', []):
                if 'box2d' not in obj:
                    continue
                category = obj['category'].split('/')[0]
                if category not in self._class_map:
                    continue
                box = obj['box2d']
                xc = (box['x1'] + box['x2']) / 2 / iw
                yc = (box['y1'] + box['y2']) / 2 / ih
                bw = (box['x2'] - box['x1']) / iw
                bh = (box['y2'] - box['y1']) / ih
                lines.append(f"{self._class_map[category]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text('\n'.join(lines), encoding='utf-8')

        # 拷贝图片
        dst_img = self._output_dir / 'images' / split / img_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

    # ------------------------------------------------------------------
    def _generate_yaml(self) -> None:
        """生成 data.yaml 供 YOLO 训练使用。"""
        nc = len(self._class_map)
        names = list(self._class_map.keys())
        yaml_text = (
            f"path: {self._output_dir}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"\nnc: {nc}\n"
            f"names: {names}\n"
        )
        (self._output_dir / 'data.yaml').write_text(yaml_text, encoding='utf-8')
        logger.info("Generated data.yaml (nc=%d)", nc)


# ===================================================================
#  CLI 入口
# ===================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='BDD100K → YOLO 格式数据集转换工具'
    )
    parser.add_argument('--images', required=True, help='BDD100K 图片根目录')
    parser.add_argument('--labels', required=True, help='BDD100K 标注根目录')
    parser.add_argument('--output', default='./bdd_yolo', help='输出目录 (默认: ./bdd_yolo)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='要处理的 split (默认: train val test)')
    parser.add_argument('--ext', default='.jpg', help='图片扩展名 (默认: .jpg)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    converter = BDD2YOLOConverter(
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        output_dir=Path(args.output),
        splits=tuple(args.splits),
        image_ext=args.ext,
    )
    converter.run()


if __name__ == '__main__':
    main()
