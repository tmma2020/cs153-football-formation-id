# CS153 Football Formation Detection

This repository is for the class CS153 at Harvey Mudd, a computer vision class. The goal of this project is to detect formations in DIII football film, hopefully reducing the amount of time it takes to manually do so. The data for this project is from Pomona Pitzer's football program, so it is not publicly available. The sample data is used to demonstrate functionality.

## Features

- Utilize yard lines to automatically crop the field from the image. Helps to reduce noise and improve the model's performance.
- Detect players using a YOLOv8 model. We use a pre-trained YOLOv8n model to detect players in the image. The model is trained on the COCO dataset, which includes person detection.
- Detect formations using a custom model. The model is trained on the sampled data to detect and classify offensive formations based on player positions.
- Sampled data is organized in a structured format. The data is organized by formation and play, making it easy to access and use for training and testing the model.

## Installation / Requirements

1. Clone the repository

```bash
   git clone https://github.com/tmma2020/cs153-football-formation-id.git
   cd cs153-football-formation-id
```

2. Install python dependencies

```bash
   pip install -r requirements.txt
```

## How to Run

1. **Organize your data**: The data should be organized in the following format:

   ```
   data/
       dataset.json
       [off_formation]/
           [video_path]/
               sideline_[video_path].png
               endzone_[video_path].png
               tight_[video_path].png
   ```

   - `dataset.json` is a metadata file containing the sampled plays. It should be in the same directory as the formation folders.
   - Each formation folder should contain a subfolder for each video path. Each video path folder should contain the images for that play.
   - The images should be named `sideline_[video_path].png`, `endzone_[video_path].png`, and `tight_[video_path].png`.

1a. **Sample data (Optional)**: The sample data is organized in the same format as above. The sample data is used to demonstrate functionality and is not a complete dataset. The sample data is located in the `sample_data/` folder.

2. **Run the field lines detection notebook**: The field lines detection notebook will detect the field lines and crop the field from the image. This is done using OpenCV and a custom algorithm to detect the yard lines.

3. **Run the player detection notebook**: The player detection notebook will detect players in the image using a YOLOv8 model. The model is trained on the COCO dataset, which includes person detection.

4. **Run the formation detection notebook**: The formation detection notebook will detect and classify offensive formations based on player positions. The model is trained on the data to detect and classify formations.

## Folder Structure

<details>
<summary><strong>Folder Structure</strong></summary>

- `README.md` — This file

- `requirements.txt` — Python dependencies

- `field_lines_detection/` — Code for detecting field lines. Used to crop the field from the image

  - `field_lines_detector.ipynb` — Notebook for detecting field lines and cropping the field

- `formation_detector/` — Code for detecting football formations from player positions

  - `formation_detector.ipynb` — Notebook for detecting and classifying offensive formations

- `sample_data/` — Example data used to demonstrate functionality

  - `sample_data_collector.ipynb` — Jupyter notebook to collect and organize sample data
  - `sample_dataset.json` — Metadata file containing the sampled plays
  - `[off_formation]/` — Folder for each sampled offensive formation (e.g., `ACES`, `KINGSSPLIT`, `QUEENS`)
    - `[video_path]/` — Folder containing images for each play
      - `sideline_[video_path].png`
      - `endzone_[video_path].png`
      - `tight_[video_path].png`

- `yolo_player_detector/` — Code and outputs for player detection
  - `player_detector.ipynb` — Notebook to detect players using a YOLOv8 model
  - `player_bboxes.json` — Detected player bounding boxes for each play
  - `yolov8n.pt` — Pre-trained YOLOv8n model weights (used for person detection)

</details>

## Notes

- Full datasets are hosted separately on the teapot server: `/cs/cs153/data/toms_project_data/`.
- The models for this project were trained on the teapot server.
