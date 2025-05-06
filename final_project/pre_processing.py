import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
RAW_DATA = "/cs/cs153/data/toms_project_data/hudl_dataset"

def detect_and_draw_lines(image):
    canny_thresh1 = 30
    canny_thresh2 = 150
    hough_thresh = 100
    min_line_length = 150
    max_line_gap = 20
    min_length_filter = 50
    angle_range_filter = (30, 150)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=canny_thresh1, threshold2=canny_thresh2)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    filtered_lines = []
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            if length >= min_length_filter and (angle_range_filter[0] <= angle <= angle_range_filter[1]):
                filtered_lines.append((x1, y1, x2, y2))

    return filtered_lines

def apply_field_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([10, 40, 40])
    upper_green = np.array([110, 255, 255])

    field_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)

    masked_image = cv2.bitwise_and(image, image, mask=field_mask)

    return masked_image

def crop_image_to_field_region(image, filtered_lines, padding=10, black_thresh=50):
    if not filtered_lines:
        return None, (0, image.shape[0])

    y_coords = []
    for (x1, y1, x2, y2) in filtered_lines:
        y_coords.append(y1)
        y_coords.append(y2)

    top_y = max(min(y_coords) - padding, 0)
    bottom_y = min(max(y_coords) + padding, image.shape[0])
    cropped = image[int(top_y):int(bottom_y), :]

    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_cropped, 50, 255, cv2.THRESH_BINARY_INV)
    height, width = binary.shape
    black_pixel_thresh = (black_thresh / 100.0) * width

    new_top = 0
    for row in range(height):
        if np.sum(binary[row, :] > 0) < black_pixel_thresh:
            new_top = row
            break
    new_bottom = height
    for row in range(height - 1, -1, -1):
        if np.sum(binary[row, :] > 0) < black_pixel_thresh:
            new_bottom = row
            break

    final_cropped = cropped[new_top:new_bottom, :]
    return final_cropped, (top_y + new_top, top_y + new_bottom)

def preprocess_and_crop():
    angle = "sideline"
    output_suffix = "cropped_sideline"

    for formation_folder in tqdm(os.listdir(RAW_DATA), desc="Formations"):
        formation_path = os.path.join(RAW_DATA, formation_folder)
        if not os.path.isdir(formation_path):
            continue

        for video_folder in os.listdir(formation_path):
            video_path = os.path.join(formation_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            sideline_image_path = os.path.join(video_path, f"{angle}_{video_folder}.png")
            if not os.path.exists(sideline_image_path):
                continue

            original_image = cv2.imread(sideline_image_path)
            if original_image is None:
                continue

            masked_image = apply_field_mask(original_image)
            masked_lines = detect_and_draw_lines(masked_image)

            final_cropped_color = original_image

            if masked_lines:
                try:
                    cropped_masked, (crop_start_y, crop_end_y) = crop_image_to_field_region(
                        masked_image, masked_lines, padding=20, black_thresh=50
                    )
                    if crop_end_y > crop_start_y and (crop_end_y - crop_start_y) > 10:
                        final_cropped_color = original_image[int(crop_start_y):int(crop_end_y), :]
                except Exception:
                    pass  # fallback to original image if error

            output_filename = f"{output_suffix}_{video_folder}.png"
            output_path = os.path.join(video_path, output_filename)
            cv2.imwrite(output_path, final_cropped_color)

if __name__ == "__main__":
    preprocess_and_crop()
