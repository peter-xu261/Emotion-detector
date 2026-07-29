#!/usr/bin/env python3
"""
Live face detection plus emotion classification for NVIDIA Jetson.

Pipeline:
    camera frame
    -> detectNet finds the largest face
    -> CUDA crops the face
    -> custom ResNet-18 imageNet model predicts emotion
    -> result is drawn on the live video
"""

import argparse
import collections
import os
import sys
from typing import Deque, Dict, Optional, Tuple

from jetson_inference import detectNet, imageNet
from jetson_utils import (
    Log,
    cudaAllocMapped,
    cudaCrop,
    cudaFont,
    videoOutput,
    videoSource,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live face detection and emotion classification on Jetson",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=detectNet.Usage() + imageNet.Usage() + videoSource.Usage()
        + videoOutput.Usage() + Log.Usage(),
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="/dev/video0",
        help="camera or input stream URI",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="display://0",
        help="display, video, or stream output URI",
    )

    parser.add_argument("--model", required=True, help="emotion ONNX model")
    parser.add_argument("--labels", required=True, help="emotion labels.txt")
    parser.add_argument("--input-blob", default="input_0")
    parser.add_argument("--output-blob", default="output_0")

    parser.add_argument(
        "--face-network",
        default="facenet",
        help="built-in face detector, use facedetect or facenet",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.45,
        help="minimum face detection confidence",
    )
    parser.add_argument(
        "--emotion-threshold",
        type=float,
        default=0.20,
        help="minimum emotion confidence shown as a prediction",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.12,
        help="fractional padding added around the detected face",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=5,
        help="number of recent predictions used to reduce flicker",
    )
    parser.add_argument(
        "--classify-every",
        type=int,
        default=1,
        help="classify once every N frames",
    )

    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model):
        parser.error(f"model file was not found: {args.model}")
    if not os.path.isfile(args.labels):
        parser.error(f"labels file was not found: {args.labels}")
    if args.smoothing < 1:
        parser.error("--smoothing must be at least 1")
    if args.classify_every < 1:
        parser.error("--classify-every must be at least 1")
    if args.padding < 0:
        parser.error("--padding must be 0 or greater")

    return args


def largest_face(detections):
    """Return the detection with the largest bounding-box area."""
    if not detections:
        return None

    return max(
        detections,
        key=lambda d: max(0.0, d.Right - d.Left)
        * max(0.0, d.Bottom - d.Top),
    )


def padded_roi(
    detection,
    image_width: int,
    image_height: int,
    padding: float,
):
    face_width = max(1.0, detection.Right - detection.Left)
    face_height = max(1.0, detection.Bottom - detection.Top)

    center_x = (detection.Left + detection.Right) / 2.0
    center_y = (detection.Top + detection.Bottom) / 2.0

    size = max(face_width, face_height)
    size *= 1.0 + 2.0 * padding

    half = size / 2.0

    left = int(center_x - half)
    top = int(center_y - half)
    right = int(center_x + half)
    bottom = int(center_y + half)

    if left < 0:
        right -= left
        left = 0

    if top < 0:
        bottom -= top
        top = 0

    if right > image_width:
        left -= right - image_width
        right = image_width

    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right - left < 20 or bottom - top < 20:
        return None

    return left, top, right, bottom


def smoothed_prediction(
    history: Deque[Tuple[str, float]],
) -> Tuple[str, float]:
    """
    Choose a stable label using confidence-weighted voting.

    The returned confidence is the average confidence for the winning label.
    """
    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for label, confidence in history:
        scores[label] = scores.get(label, 0.0) + confidence
        counts[label] = counts.get(label, 0) + 1

    label = max(scores, key=scores.get)
    confidence = scores[label] / counts[label]
    return label, confidence


def main() -> int:
    args = parse_args()

    face_net = detectNet(
        args.face_network,
        threshold=args.face_threshold,
    )

    emotion_net = imageNet(
        model=args.model,
        labels=args.labels,
        input_blob=args.input_blob,
        output_blob=args.output_blob,
    )

    camera = videoSource(args.input, argv=sys.argv)
    display = videoOutput(args.output, argv=sys.argv)
    font = cudaFont()

    history: Deque[Tuple[str, float]] = collections.deque(
        maxlen=args.smoothing
    )

    crop = None
    crop_shape = None
    frame_number = 0
    current_label = "No face"
    current_confidence = 0.0

    while True:
        image = camera.Capture()

        if image is None:
            if not camera.IsStreaming():
                break
            continue

        frame_number += 1

        # detectNet draws the face box directly onto the camera image
        detections = face_net.Detect(image, overlay="none")
        face = largest_face(detections)

        if face is None:
            history.clear()
            current_label = "No face"
            current_confidence = 0.0
        else:
            roi = padded_roi(
                face,
                image.width,
                image.height,
                args.padding,
            )

            if roi is not None and frame_number % args.classify_every == 0:
                left, top, right, bottom = roi
                crop_width = right - left
                crop_height = bottom - top
                required_shape = (
                    crop_width,
                    crop_height,
                    image.format,
                )

                # Reuse the CUDA crop buffer until the box size changes.
                if crop is None or crop_shape != required_shape:
                    crop = cudaAllocMapped(
                        width=crop_width,
                        height=crop_height,
                        format=image.format,
                    )
                    crop_shape = required_shape

                cudaCrop(
                    input=image,
                    output=crop,
                    roi=(left, top, right, bottom),
                )

                class_id, confidence = emotion_net.Classify(crop)
                label = emotion_net.GetClassLabel(class_id)

                history.append((label, confidence))
                current_label, current_confidence = smoothed_prediction(
                    history
                )

            if roi is not None:
                left, top, _, _ = roi
                text_y = max(5, top - font.GetSize() - 8)
            else:
                left, text_y = 5, 5

            if current_confidence >= args.emotion_threshold:
                text = (
                    f"{current_label}: "
                    f"{current_confidence * 100.0:.1f}%"
                )
            else:
                text = "Emotion uncertain"

            font.OverlayText(
                image,
                text=text,
                x=max(5, left),
                y=text_y,
                color=font.White,
                background=font.Gray40,
            )

        display.Render(image)
        # Avoid querying GetNetworkFPS() here.  The two networks run
        # asynchronously, and querying their CUDA timing events every frame can
        # produce repeated CUDA error 600 (operation not ready) messages.
        display.SetStatus(
            f"Emotion Detector | {current_label} "
            f"{current_confidence * 100.0:.1f}%"
        )

        if not camera.IsStreaming() or not display.IsStreaming():
            break

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")