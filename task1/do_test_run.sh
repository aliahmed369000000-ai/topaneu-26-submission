#!/usr/bin/env bash
# اختبار محلي لحالة واحدة عبر Docker (محاكاة Grand Challenge)
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-topaneu-task1}"
TEST_DIR="${TEST_DIR:-./test}"
SCRIPT_DIR="$(pwd)"

echo "==> بناء الصورة (إن لزم)"
docker build -t "$IMAGE_NAME" .

mkdir -p "$TEST_DIR/input" "$TEST_DIR/output" "$TEST_DIR/model"

# الأوزان: من ./models أو من MODEL_TAR
if [[ -f "${MODEL_TAR:-}" ]]; then
  echo "==> استخراج Model tarball إلى test/model"
  tar -xzf "$MODEL_TAR" -C "$TEST_DIR/model"
elif ls models/topaneu_task1_fold*.pt >/dev/null 2>&1; then
  echo "==> نسخ الأوزان من models/"
  cp -f models/topaneu_task1_fold*.pt "$TEST_DIR/model/" 2>/dev/null || true
  cp -f models/thresholds.json "$TEST_DIR/model/" 2>/dev/null || true
else
  echo "⚠ ضع الأوزان في task1/models/ أو مرّر MODEL_TAR=topaneu_task1_model.tar.gz"
fi

if [[ ! -f "$TEST_DIR/input/inputs.json" ]]; then
  echo "خطأ: أنشئ بيانات الاختبار أولاً:"
  echo "  python prepare_test_case.py --image /path/to/case.nii.gz --modality ct"
  exit 1
fi

echo "==> docker run"
docker run --rm \
  --gpus all \
  -v "$(realpath "$TEST_DIR/input")":/input:ro \
  -v "$(realpath "$TEST_DIR/output")":/output \
  -v "$(realpath "$TEST_DIR/model")":/opt/ml/model:ro \
  "$IMAGE_NAME"

echo "==> النتيجة:"
cat "$TEST_DIR/output/detected-aneurysm-locations.json"
echo
