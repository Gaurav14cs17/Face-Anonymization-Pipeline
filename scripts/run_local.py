#!/usr/bin/env python3
"""Run anonymization and QC on one local video."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s23_anonymizer import FaceAnonymizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Anonymize video with YuNet + QC")
    parser.add_argument("input", type=Path, help="Input .mp4 path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: output/<stem>_anonymized.mp4)",
    )
    parser.add_argument("--no-tracker", action="store_true")
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="Draw visible face bounding boxes on the output video",
    )
    args = parser.parse_args()

    out = args.output or ROOT / "output" / (args.input.stem + "_anonymized.mp4")
    yunet = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    blaze = ROOT / "models" / "blaze_face_short_range.tflite"

    anonymizer = FaceAnonymizer(
        yunet,
        blaze,
        draw_boxes=args.draw_boxes,
        use_tracker=not args.no_tracker,
    )
    summary = anonymizer.process(args.input, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
