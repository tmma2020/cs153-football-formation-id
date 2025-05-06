# This script is just a compilation of the jupyter notebooks show in my Github. the paths are relative to the server structure I was using

import os, json, cv2, numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Paths
RAW_DATA = "/cs/cs153/data/toms_project_data/hudl_dataset"
PLAYER_JSON = "player_bboxes.json"
DATASET_JSON = "/cs/cs153/data/toms_project_data/hudl_dataset/dataset.json"
OUTPUT_JSON = PLAYER_JSON

def run_yolo_detection():
    model = YOLO("yolov8x.pt")
    detections = []

    cropped_image_paths = []

    for formation_folder in os.listdir(RAW_DATA):
        formation_path = os.path.join(RAW_DATA, formation_folder)
        if not os.path.isdir(formation_path):
            continue

        for video_folder in os.listdir(formation_path):
            video_path = os.path.join(formation_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            for file in os.listdir(video_path):
                if file.startswith("cropped_sideline") and file.endswith(".png"):
                    full_path = os.path.join(video_path, file)
                    cropped_image_paths.append(full_path)

    # run YOLO on all cropped images
    for img_path in tqdm(cropped_image_paths, desc="Running YOLO Detection"):
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

    # save all detections
    with open(OUTPUT_JSON, "w") as f:
        json.dump(detections, f, indent=4)


def classify_formations():
    # load detections
    with open(PLAYER_JSON, "r") as f:
        all_detections = json.load(f)

    # load metadata
    with open(DATASET_JSON, "r") as f:
        dataset_metadata = [json.loads(line) for line in f]

    # map video to formation
    video_to_formation = {
        entry["video_path"]: entry["off_formation"]
        for entry in dataset_metadata
    }

    X, y = [], []

    for detection in all_detections:
        filename = detection["image_filename"]
        video_path = filename.replace("cropped_sideline_", "").replace(".png", "")
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

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # visualize samples
    num_samples_to_show = 5
    demo_indices = np.random.choice(len(X_test), size=min(num_samples_to_show, len(X_test)), replace=False)

    for idx in demo_indices:
        sample = X_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        matching_index = np.where((X == sample).all(axis=1))[0]
        if len(matching_index) == 0:
            continue
        matching_index = matching_index[0]

        detection = all_detections[matching_index]
        filename = detection["image_filename"]

        found_path = None
        for root, dirs, files in os.walk(RAW_DATA):
            if filename in files:
                found_path = os.path.join(root, filename)
                break

        if found_path is None:
            continue

        image = cv2.imread(found_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for player in detection["players"]:
            x1, y1, x2, y2 = player["bbox"]
            cv2.rectangle(
                image_rgb,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=(0, 255, 0),
                thickness=3
            )

        label_text = f"Predicted: {pred_label}"
        true_text = f"True: {true_label}"

        cv2.putText(image_rgb, label_text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 215, 0), 3)
        cv2.putText(image_rgb, true_text, (10, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (173, 216, 230), 3)

        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title("Formation Prediction Visualization", fontsize=18)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_yolo_detection()
    classify_formations()
