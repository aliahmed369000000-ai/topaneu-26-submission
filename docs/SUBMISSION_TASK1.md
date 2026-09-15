# تسليم TopAneu-26 Task 1

## المحتويات الجاهزة

1. **الكود** في `task1/`: `main.py`, `inference.py`, `model.py`, `vessel_preprocessing.py`, `Dockerfile`
2. **الأوزان + thresholds** كملف:
   - محلياً بعد التدريب: `task1/models/topaneu_task1_fold{0-4}.pt` + `thresholds.json`
   - أو ارفع tarball: `topaneu_task1_model.tar.gz` (يُستخرج إلى `/opt/ml/model` على Grand Challenge)

## إعدادات الاستدلال

- Ensemble: 5 folds
- TTA flip: مفعّل
- Threshold: `global_threshold=0.15` من معايرة OOF (يفضَّل على per-class حسب الـ score)

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
