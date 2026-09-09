# topaneu-26-submission

بنية تحتية لتقديم Task 1 (تصنيف Multi-label) في تحدي TopAneu 2026.

## الحالة الحالية (محدّثة 2026-09-09)

- ✅ معالجة قناع الأوعية (Nearest-Neighbor + Binary): `task1/vessel_preprocessing.py`
- ✅ تقسيم K-Fold متعدد الفئات (Multilabel Stratified): `task1/dataset_split.py`
- ✅ المعمارية الكاملة (Backbone v1 + رأس Skip Connections، feature_dim=512،
  ~3.14M معامل): `task1/model.py`
- ✅ حلقة تدريب كاملة (K-Fold + pos_weight + Gradient Clipping + Flip
  Augmentation): `task1/train.py`
- ✅ **inference.py متوافق 100% مع القالب الرسمي** (`infer_ct` / `infer_mr`
  يأخذان `sitk.Image` ويرجعان `list[int]`) + Ensemble + TTA Flip
- ✅ `main.py` من القالب الرسمي
- ✅ `Dockerfile` متوافق مع هيكل Grand Challenge
- ✅ `calibrate_threshold.py` لمعايرة threshold على Precision/Recall/MCC
- ❌ لم يُدرَّب نموذج بعد — `task1/models/` فارغ، شغّل `train.py` على
  بيانات حقيقية ثم `calibrate_threshold.py` قبل بناء الحاوية

## خطوات التشغيل

```bash
# 1. تدريب
python train.py --data-root /path/to/topaneu --output-dir ./models --epochs 30

# 2. معايرة threshold (مقاييس رسمية)
python calibrate_threshold.py --data-root /path/to/topaneu --models-dir ./models

# 3. حدّث DECISION_THRESHOLD في inference.py بالرقم الناتج

# 4. بناء واختبار الحاوية (من مجلد task1)
bash do_build.sh   # إن وُجد، أو docker build ...
```

## ملاحظات مهمة للفوز

- المقاييس الرسمية: **Precision + Recall + MCC** (ليس AUC).
- لا يوجد vessel mask في الاستدلال → النموذج يستخدم قناة صفرية مؤقتًا.
- Ensemble من 5 folds + Test-Time Flip مفعّل في `inference.py`.
- threshold الافتراضي 0.45 (يُفضّل معايرته).

## البناء التلقائي

عند أي push لمسار `task1/`، تعمل GitHub Action تلقائيًا لبناء الحاوية
ورفعها كـ Artifact قابل للتحميل.
