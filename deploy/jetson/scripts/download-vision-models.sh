#!/usr/bin/env bash
# Download ONNX models for vision-trt service.
# Run from anywhere; models are saved to deploy/jetson/vision-trt/models/.
#
# Sources (all public domain / permissive licenses):
#   SCRFD-10G: InsightFace antelopev2 release (insightface.ai)
#   ArcFace MobileFaceNet: InsightFace buffalo_s release (insightface.ai)
#   FER+ EfficientNet: ONNX Model Zoo (onnx/models, MIT/Apache license)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../vision-trt/models"
mkdir -p "$MODELS_DIR"

echo "Downloading vision-trt ONNX models to $MODELS_DIR ..."

# ── SCRFD-10G face detection (with 5 keypoints) ────────────────────────────────
if [ ! -f "$MODELS_DIR/scrfd_10g_bnkps.onnx" ]; then
    echo "Downloading scrfd_10g_bnkps.onnx (~17MB)..."
    TMP=$(mktemp -d)
    curl -L -o "$TMP/antelopev2.zip" \
        "https://drive.google.com/uc?export=download&id=18pAVESzW4WNa3HnRJhEhqMQ-r7BqNtNv&confirm=t"
    unzip -j "$TMP/antelopev2.zip" "*/scrfd_10g_bnkps.onnx" -d "$MODELS_DIR"
    rm -rf "$TMP"
    echo "  scrfd_10g_bnkps.onnx done"
else
    echo "  scrfd_10g_bnkps.onnx already exists, skipping"
fi

# ── ArcFace MobileFaceNet (face recognition embedding) ────────────────────────
if [ ! -f "$MODELS_DIR/w600k_mbf.onnx" ]; then
    echo "Downloading w600k_mbf.onnx (~13MB)..."
    TMP=$(mktemp -d)
    curl -L -o "$TMP/buffalo_s.zip" \
        "https://drive.google.com/uc?export=download&id=1l7JDqALKDPFAlIIhR8Q2VXFR2IFsLBSV&confirm=t"
    unzip -j "$TMP/buffalo_s.zip" "*/w600k_mbf.onnx" -d "$MODELS_DIR"
    rm -rf "$TMP"
    echo "  w600k_mbf.onnx done"
else
    echo "  w600k_mbf.onnx already exists, skipping"
fi

# ── FER+ emotion recognition (ONNX Model Zoo, public) ─────────────────────────
if [ ! -f "$MODELS_DIR/emotion-ferplus-8.onnx" ]; then
    echo "Downloading emotion-ferplus-8.onnx (~34MB)..."
    curl -L -o "$MODELS_DIR/emotion-ferplus-8.onnx" \
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    echo "  emotion-ferplus-8.onnx done"
else
    echo "  emotion-ferplus-8.onnx already exists, skipping"
fi

echo ""
echo "All models ready in $MODELS_DIR"
ls -lh "$MODELS_DIR"
