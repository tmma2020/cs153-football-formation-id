import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime

# Paths
RAW_DATA = "/cs/cs153/data/toms_project_data/hudl_dataset"
PLAYER_JSON = "player_bboxes_raw.json"  # <-- Different output so it doesn't overwrite
DATASET_JSON = "/cs/cs153/data/toms_project_data/hudl_dataset/dataset.json"
OUTPUT_JSON = PLAYER_JSON

def run_yolo_detection():
    model = YOLO("yolov8x.pt")
    detections = []

    sideline_image_paths = []

    for formation_folder in os.listdir(RAW_DATA):
        formation_path = os.path.join(RAW_DATA, formation_folder)
        if not os.path.isdir(formation_path):
            continue

        for video_folder in os.listdir(formation_path):
            video_path = os.path.join(formation_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            # Now looking for raw sideline images
            sideline_filename = f"sideline_{video_folder}.png"
            sideline_path = os.path.join(video_path, sideline_filename)

            if os.path.exists(sideline_path):
                sideline_image_paths.append(sideline_path)

    # Run YOLO on all raw sideline images
    for img_path in tqdm(sideline_image_paths, desc="Running YOLO Detection (Raw Images)"):
        results = model(img_path, conf=0.25)
        result = results[0]

        player_data = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box

            if int(cls) == 0:  # 'person' class
                player_data.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        detections.append({
            "image_filename": os.path.basename(img_path),
            "players": player_data
        })

    # Save all detections
    with open(OUTPUT_JSON, "w") as f:
        json.dump(detections, f, indent=4)

def classify_formations():
    # Load detections
    with open(PLAYER_JSON, "r") as f:
        all_detections = json.load(f)

    # Load metadata
    with open(DATASET_JSON, "r") as f:
        dataset_metadata = [json.loads(line) for line in f]

    # Map video to formation
    video_to_formation = {
        entry["video_path"]: entry["off_formation"]
        for entry in dataset_metadata
    }

    X, y = [], []

    for detection in all_detections:
        filename = detection["image_filename"]
        video_path = filename.replace("sideline_", "").replace(".png", "")
        formation_label = video_to_formation.get(video_path)

        if formation_label is None:
            continue

        player_positions = []
        for player in detection["players"]:
            x1, y1, x2, y2 = player["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            player_positions.append([cx, cy])

        flattened = np.array(player_positions).flatten()
        max_players = 20

        if len(flattened) < max_players * 2:
            flattened = np.pad(flattened, (0, max_players * 2 - len(flattened)))
        else:
            flattened = flattened[:max_players * 2]

        X.append(flattened)
        y.append(formation_label)

    X = np.array(X)
    y = np.array(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "results_processed"  # or "results_raw" depending on the file
    os.makedirs(save_dir, exist_ok=True)

    report = classification_report(y_test, y_pred)
    print(report)

    # Save to file
    with open(os.path.join(save_dir, f"classification_raw_report_{timestamp}.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    run_yolo_detection()
    classify_formations()
