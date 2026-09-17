# تسليم TopAneu-26 Task 1

## المحتويات الجاهزة

1. **الكود** في `task1/`: `main.py`, `inference.py`, `model.py`, `vessel_preprocessing.py`, `Dockerfile`
2. **الأوزان + thresholds** كملف:
   - محلياً بعد التدريب: `task1/models/topaneu_task1_fold{0-4}.pt` + `thresholds.json`
   - أو ارفع tarball: `topaneu_task1_model.tar.gz` (يُستخرج إلى `/opt/ml/model` على Grand Challenge)

## إعدادات الاستدلال

- Ensemble: 5 folds
- TTA flip: مفعّل
- Threshold: `global_threshold=0.55` من فحص OOF (415 حالة) — 0.15 كان يعظّم Score إحصائيًا لكن ينتج ~21 تنبؤًا/حالة (غير واقعي طبيًا مقابل ~0.95 حقيقي)؛ 0.55 يوازن Score مع واقعية عدد المواقع (mean≈2.5, median=2)

## خطوات Grand Challenge

1. ابنِ صورة Docker من مجلد `task1/` (بدون الحاجة لوضع `.pt` داخل الصورة إذا رفعت Model منفصلاً):
   ```bash
   cd task1
   docker build -t topaneu-task1 .
   ```
2. ارفع الـ Algorithm على https://topaneu-26.grand-challenge.org
3. ارفع **Model** = `topaneu_task1_model.tar.gz` (المحتوى يظهر في `/opt/ml/model`)
4. اربط الـ Model بالـ Algorithm
5. اختبر على try-out ثم أرسل للتقييم الرسمي

## قيود المنصة

- ≤ 12 دقيقة / حالة
- GPU T4 16GB
- RAM 32GB

## ملاحظة

النموذج مدرّب على قناتين؛ وقت الاستدلال لا يوجد vessel mask → قناة ثانية صفرية.

## اختبار محلي بـ Docker (حالة واحدة)

```bash
cd task1

# 1) حضّر حالة (NIfTI من بيانات التدريب مثلاً)
python prepare_test_case.py \
  --image /path/to/topaneu_center2_ct_105_0000.nii.gz \
  --modality ct \
  --out-dir ./test

# 2) الأوزان: إمّا داخل models/ أو tarball
#    انسخ topaneu_task1_fold*.pt + thresholds.json إلى models/
#    أو:
#    export MODEL_TAR=/path/to/topaneu_task1_model.tar.gz

# 3) بناء + تشغيل (GPU إن وُجد؛ احذف --gpus all إن لم يتوفر)
./do_test_run.sh

# 4) اقرأ النتيجة
cat test/output/detected-aneurysm-locations.json
```

تشغيل يدوي مكافئ:

```bash
docker build -t topaneu-task1 .
docker run --rm \
  -v $PWD/test/input:/input:ro \
  -v $PWD/test/output:/output \
  -v $PWD/test/model:/opt/ml/model:ro \
  topaneu-task1
```

`DECISION_THRESHOLD=0.55` (مثبت في inference.py و thresholds.json).
