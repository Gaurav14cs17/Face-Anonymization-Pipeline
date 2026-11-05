#!/usr/bin/env python3
"""Verify all dependencies and setup before running pipeline."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.8."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"Python 3.8+ required, got {version.major}.{version.minor}"
    return True, f"✅ Python {version.major}.{version.minor}"


def check_directory_structure():
    """Check required directories exist."""
    root = Path(__file__).resolve().parents[1]
    required = ["src", "data", "output", "models", "scripts"]
    missing = []

    for d in required:
        if not (root / d).exists():
            missing.append(d)

    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "✅ Directory structure OK"


def check_models():
    """Check required model files exist."""
    root = Path(__file__).resolve().parents[1]
    models = {
        "YuNet": "models/face_detection_yunet_2023mar.onnx",
        "BlazeFace": "models/blaze_face_short_range.tflite",
    }
    missing = []

    for name, path in models.items():
        if not (root / path).exists():
            missing.append(f"{name} ({path})")

    if missing:
        return False, f"Missing models: {', '.join(missing)}"
    return True, "✅ All models present"


def check_sample_video():
    """Check sample video exists."""
    root = Path(__file__).resolve().parents[1]
    video = root / "data" / "samples" / "youtube_-Koj9hvcBMk.mp4"

    if not video.exists():
        return False, "Sample video not found: data/samples/youtube_-Koj9hvcBMk.mp4"
    return True, f"✅ Sample video ({video.stat().st_size / 1e6:.1f} MB)"


def check_dependencies():
    """Check Python packages."""
    required = {
        "cv2": "opencv-contrib-python",
        "fastapi": "fastapi",
        "mediapipe": "mediapipe",
    }
    missing = []

    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{package}")

    if missing:
        return False, f"Missing packages: {', '.join(missing)}\nRun: pip install -r requirements.txt"
    return True, "✅ All dependencies installed"


def check_python_files():
    """Check all required Python files exist."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"

    required = [
        "s02_detection.py",
        "s02_blur.py",
        "s03_tracker.py",
        "s04_jobs.py",
        "s04_app.py",
        "s05_verify.py",
        "s23_anonymizer.py",
    ]
    missing = []

    for f in required:
        if not (src / f).exists():
            missing.append(f)

    if missing:
        return False, f"Missing source files: {', '.join(missing)}"
    return True, "✅ All source files present"


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("SETUP VERIFICATION")
    print("=" * 70 + "\n")

    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Model Files", check_models),
        ("Sample Video", check_sample_video),
        ("Python Dependencies", check_dependencies),
        ("Source Files", check_python_files),
    ]

    results = []
    for name, check_func in checks:
        ok, msg = check_func()
        results.append((ok, name, msg))
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}: {name}")
        print(f"       {msg}\n")

    # Summary
    passed = sum(1 for ok, _, _ in results if ok)
    total = len(results)

    print("=" * 70)
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\nReady to run:")
        print("  python scripts/run_local.py data/samples/youtube_-Koj9hvcBMk.mp4")
        print("  cd src && uvicorn s04_app:app --reload")
        print("=" * 70 + "\n")
        return 0
    else:
        print(f"❌ SETUP INCOMPLETE ({passed}/{total} checks passed)")
        print("\nPlease fix the issues above and try again.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
