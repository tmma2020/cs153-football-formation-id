# CS153 Football Formation Detection

This repository is for the class CS153 at Harvey Mudd, a computer vision class. The goal of this project is to detect formations in DIII football film, hopefully reducing the amount of time it takes to manually do so. The data for this project is from Pomona Pitzer's football program, so it is not publicly available. The sample data is used to demonstrate functionality.

## Features

-

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

## Folder Structure

## Folder Structure

<details>
<summary><strong>📁 Folder Structure (click to expand)</strong></summary>

- `sample_data/` — Example data used to demonstrate functionality
  - `sample_dataset.json` — Metadata file containing the sampled plays
  - `[off_formation]/` — Folder for each sampled offensive formation (e.g., `ACES`, `KINGSSPLIT`, `QUEENS`)
    - `[video_path]/` — Folder containing images for each play
      - `sideline_[video_path].png`
      - `endzone_[video_path].png`
      - `tight_[video_path].png`

</details>

## Notes

- Full datasets are hosted separately on the teapot server: `/cs/cs153/data/toms_project_data/`.
- The models for this project were trained on the teapot server.
