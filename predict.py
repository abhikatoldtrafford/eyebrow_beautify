"""
Eyebrow Detection and Segmentation Prediction Script
Uses YOLO11 segmentation model to detect and segment:
- eyes
- eye_box
- eyebrows
- hair
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Try to import MediaPipe (optional)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, RuntimeError):
    MEDIAPIPE_AVAILABLE = False


def load_model(model_path):
    """Load the YOLO11 segmentation model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model


def get_mediapipe_landmarks(img):
    """Extract face landmarks using MediaPipe."""
    if not MEDIAPIPE_AVAILABLE:
        return None

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                return None

            return results.multi_face_landmarks[0]
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return None


def draw_face_landmarks(img, landmarks, draw_eyebrows=True, draw_eyes=True, draw_all=True):
    """Draw MediaPipe face landmarks on image."""
    if landmarks is None:
        return img

    h, w = img.shape[:2]

    # Define landmark indices for key features
    # Left eyebrow: 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    # Right eyebrow: 300, 293, 334, 296, 336, 285, 295, 282, 283, 276
    left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_eyebrow_indices = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

    # Left eye: 33, 133, 160, 159, 158, 157, 173, 246
    # Right eye: 362, 263, 387, 386, 385, 384, 398, 466
    left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
    right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]

    # Face oval: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Lips outer: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    lips_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    # Nose tip: 1, 2, 98, 327
    nose_indices = [1, 2, 98, 327]

    # Draw all 468 landmarks as small dots
    if draw_all:
        for i, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Draw all landmarks in light gray
            cv2.circle(img, (x, y), 1, (150, 150, 150), -1)

    # Draw face oval points (no lines)
    for idx in face_oval_indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Green points

    # Draw lips points (no lines)
    for idx in lips_outer_indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(img, (x, y), 2, (255, 100, 100), -1)  # Light blue points

    # Draw nose points
    for idx in nose_indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(img, (x, y), 3, (0, 165, 255), -1)  # Orange points

    # Draw eyebrow landmarks as points (emphasized, no lines)
    if draw_eyebrows:
        for idx in left_eyebrow_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 4, (255, 0, 255), -1)  # Magenta points (larger)

        for idx in right_eyebrow_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 4, (255, 0, 255), -1)  # Magenta points (larger)

    # Draw eye landmarks as points (emphasized, no lines)
    if draw_eyes:
        for idx in left_eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)  # Yellow points

        for idx in right_eye_indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)  # Yellow points

    return img


def predict_and_visualize(model, image_path, output_path=None, conf_threshold=0.25, show=False, use_mediapipe=False):
    """
    Run prediction on an image and visualize results with bounding boxes and masks.

    Args:
        model: YOLO model
        image_path: Path to input image
        output_path: Path to save output image (optional)
        conf_threshold: Confidence threshold for predictions
        show: Whether to display the image
        use_mediapipe: Whether to overlay MediaPipe face landmarks

    Returns:
        Results object from YOLO
    """
    print(f"Running prediction on: {image_path}")

    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=0.7,
        imgsz=800,
        verbose=False
    )

    # Get the first result (single image)
    result = results[0]

    # Read original image
    img = cv2.imread(str(image_path))
    img_with_detections = img.copy()

    # Class names from the model
    class_names = model.names

    # Define colors for each class (BGR format)
    colors = {
        0: (255, 0, 0),      # eye - Blue
        1: (0, 255, 0),      # eye_box - Green
        2: (0, 0, 255),      # eyebrows - Red
        3: (255, 255, 0),    # hair - Cyan
    }

    # Check if there are any detections
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Draw segmentation masks if available
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Segmentation masks

            # Create an overlay for masks
            overlay = img_with_detections.copy()

            for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
                color = colors.get(class_id, (255, 255, 255))

                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Apply colored mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask_binary == 1] = color

                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

            # Blend overlay with original image
            img_with_detections = cv2.addWeighted(img_with_detections, 0.6, overlay, 0.4, 0)

        # Draw bounding boxes and labels
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            color = colors.get(class_id, (255, 255, 255))
            class_name = class_names[class_id]

            # Draw bounding box
            cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)

            # Draw label with confidence
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1] + 10)

            # Draw label background
            cv2.rectangle(
                img_with_detections,
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                img_with_detections,
                label,
                (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        print(f"Detected {len(boxes)} objects:")
        for class_id, conf in zip(class_ids, confidences):
            print(f"  - {class_names[class_id]}: {conf:.2f}")
    else:
        print("No detections found.")

    # Add MediaPipe landmarks if requested
    if use_mediapipe:
        if not MEDIAPIPE_AVAILABLE:
            print("Warning: MediaPipe not available, skipping landmark detection")
        else:
            print("Detecting MediaPipe face landmarks...")
            landmarks = get_mediapipe_landmarks(img)
            if landmarks:
                img_with_detections = draw_face_landmarks(img_with_detections, landmarks)
                print("✓ MediaPipe landmarks added:")
                print("  - 468 face points (gray dots - size 1)")
                print("  - Face oval points (green - size 2)")
                print("  - Eyebrow points (magenta - size 4, 10 per eyebrow)")
                print("  - Eye points (yellow - size 3, 8 per eye)")
                print("  - Lip points (light blue - size 2)")
                print("  - Nose points (orange - size 3)")
            else:
                print("✗ No face landmarks detected")

    # Save output image if path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_with_detections)
        print(f"Saved output to: {output_path}")

    # Display image if requested
    if show:
        cv2.imshow('Predictions', img_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run eyebrow detection and segmentation on an image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='eyebrow_training/eyebrow_recommended/weights/best.pt',
        help='Path to the model weights'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the output image (default: auto-generate)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the output image'
    )
    parser.add_argument(
        '--mediapipe',
        action='store_true',
        help='Overlay MediaPipe face landmarks (eyebrows and eyes)'
    )

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        input_path = Path(args.image)
        args.output = input_path.parent / f"{input_path.stem}_predicted{input_path.suffix}"

    # Load model
    model = load_model(args.model)

    # Run prediction
    predict_and_visualize(
        model,
        args.image,
        output_path=args.output,
        conf_threshold=args.conf,
        show=args.show,
        use_mediapipe=args.mediapipe
    )


if __name__ == "__main__":
    main()
