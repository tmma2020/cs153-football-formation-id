import argparse
import pre_processing
import yolo_with_processed_images as processed_pipeline
import yolo_with_raw_images as raw_pipeline
import resnet_raw
import resnet_processed

def main(run_processing, run_processed_pipeline, run_raw_pipeline, run_resnet_raw_model, run_resnet_processed_model):
    print(f"[DEBUG] run_processing={run_processing}, run_processed_pipeline={run_processed_pipeline}, run_raw_pipeline={run_raw_pipeline}, run_resnet_raw_model={run_resnet_raw_model}, run_resnet_processed_model={run_resnet_processed_model}")
    
    if run_processing:
        print("[INFO] Running preprocessing...")
        pre_processing.preprocess_and_crop()

    if run_processed_pipeline:
        print("[INFO] Running YOLO detection on processed images...")
        processed_pipeline.run_yolo_detection()

        print("[INFO] Running formation classification (processed images)...")
        processed_pipeline.classify_formations()

    if run_raw_pipeline:
        print("[INFO] Running YOLO detection on raw images...")
        raw_pipeline.run_yolo_detection()

        print("[INFO] Running formation classification (raw images)...")
        raw_pipeline.classify_formations()

    if run_resnet_raw_model:
        print("[INFO] Running ResNet classification on raw images...")
        resnet_raw.run_resnet_raw()

    if run_resnet_processed_model:
        print("[INFO] Running ResNet classification on processed images...")
        resnet_processed.run_resnet_processed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full HUDL data pipeline.")

    parser.add_argument(
        "--process",
        action="store_true",
        help="Run the preprocessing step (field detection, masking, cropping)"
    )
    parser.add_argument(
        "--processed",
        action="store_true",
        help="Run YOLO detection + classification on processed (cropped) images"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Run YOLO detection + classification on raw images"
    )
    parser.add_argument(
        "--resnet_raw",
        action="store_true",
        help="Run ResNet classification on raw images"
    )
    parser.add_argument(
        "--resnet_processed",
        action="store_true",
        help="Run ResNet classification on processed (cropped) images"
    )

    args = parser.parse_args()

    main(
        args.process,
        args.processed,
        args.raw,
        args.resnet_raw,
        args.resnet_processed
    )
