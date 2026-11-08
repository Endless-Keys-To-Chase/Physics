import csv
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# 1. Load your downloaded weights locally
model_path = "/Upload/best.pt"  # Path to your downloaded YOLOv8 weights
model = YOLO(model_path)

video_folder = "/Upload/videos"  # Folder containing your .mp4 files
output_folder = "/Upload/results"
os.makedirs(output_folder, exist_ok=True)

video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
print(f"Found {len(video_files)} videos. Starting offline batch inference...")

for video_idx, video_path in enumerate(video_files, 1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n[{video_idx}/{len(video_files)}] Processing {video_name}...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    data_log = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp = frame_index / fps

        # Perform local inference directly in VM memory
        results = model.predict(frame, verbose=False, conf=0.25)[0]

        cx, cy = 0, 0
        if len(results.boxes) > 0:
            # Pick highest confidence detection
            best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
            box_coords = best_box.xywh[0].cpu().numpy()  # Center X, Center Y, W, H
            cx, cy = int(box_coords[0]), int(box_coords[1])

        data_log.append([frame_index, round(timestamp, 4), cx, cy])
        frame_index += 1

    cap.release()

    # Save CSV
    csv_path = os.path.join(output_folder, f"{video_name}_trajectory.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Time_s", "X_coord", "Y_coord"])
        writer.writerows(data_log)

    # Plot & save Y-coordinate trajectory
    data_arr = np.array(data_log, dtype=float)
    valid_mask = data_arr[:, -1] > 0

    if np.any(valid_mask):
        times = data_arr[valid_mask, 1]
        y_coords = data_arr[valid_mask, -1]

        plt.figure(figsize=(10, 4))
        plt.plot(times, y_coords, color="navy", linewidth=1.2)
        plt.title(f"Damped Oscillation Trajectory — {video_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Y Position (pixels)")
        plt.grid(True, linestyle="--", alpha=0.6)

        plot_path = os.path.join(output_folder, f"{video_name}_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

print(
    "\nProcessing complete! Clean CSVs and trajectory plots saved to ./results"
)
