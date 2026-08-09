"""
YOLO 视频推理工具 — 对视频逐帧检测并输出带标注的视频

用法:
    python -m tools.video_inference --model best.pt --video test.mp4 --output result.mp4
"""

import time
import logging
import argparse

import cv2

logger = logging.getLogger(__name__)


# ===================================================================
#  核心
# ===================================================================

def run_video_inference(
    model_path: str,
    video_path: str,
    output_path: str | None = None,
    show: bool = True,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %s | %dx%d @ %.1f fps | %d frames",
                video_path, w, h, fps, total_frames)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        logger.info("Output: %s", output_path)

    start_t = time.time()
    frame_count = 0
    total_latency = 0.0
    window_title = 'YOLO Inference'

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            results = model(frame)
            latency = time.time() - t0
            total_latency += latency
            frame_count += 1

            annotated = results[0].plot()

            # 绘制性能信息
            elapsed = time.time() - start_t
            fps_disp = frame_count / max(elapsed, 0.001)
            lat_ms = (total_latency / frame_count) * 1000
            cv2.putText(annotated, f"FPS: {fps_disp:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Latency: {lat_ms:.1f} ms", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start_t
    logger.info("Done. %d frames in %.1fs | avg FPS: %.1f | avg latency: %.1f ms",
                frame_count, elapsed, frame_count / max(elapsed, 0.001),
                (total_latency / max(frame_count, 1)) * 1000)


# ===================================================================
#  CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description='YOLO 视频推理')
    parser.add_argument('--model', required=True, help='模型路径 (.pt)')
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--output', help='输出视频路径 (可选)')
    parser.add_argument('--no-show', action='store_true', help='不显示窗口')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_video_inference(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
