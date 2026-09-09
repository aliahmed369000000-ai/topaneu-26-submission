# topaneu-26-submission

بنية تحتية لتقديم Task 1 (تصنيف Multi-label) في تحدي TopAneu 2026.

## الحالة الحالية

- ✅ معالجة قناع الأوعية (Nearest-Neighbor + Binary): `task1/vessel_preprocessing.py`
- ✅ تقسيم K-Fold متعدد الفئات (Multilabel Stratified): `task1/dataset_split.py`
- ✅ المعمارية الكاملة (Backbone v1 + رأس Skip Connections، feature_dim=512،
  3.1M معامل، مُتحقق منها بتشغيل الكود): `task1/model.py`
- ✅ حلقة تدريب كاملة (K-Fold + pos_weight + Gradient Clipping + Flip
  Augmentation): `task1/train.py` — لم تُختبر بعد على بيانات حقيقية
- ⚠️ هيكل الاستدلال: `task1/inference.py` — مبني على حقائق مؤكدة من
  [github.com/Bangulli/TopAneu-26](https://github.com/Bangulli/TopAneu-26)
  لكن يحتاج تحقق يدوي من `main.py` الرسمي (انظر التعليقات أعلى الملف)
- ❌ لم يُدرَّب نموذج بعد — `task1/models/` فارغ، يجب تشغيل `train.py` على
  بيانات حقيقية ووضع الأوزان الناتجة قبل بناء الحاوية
- ⚠️ `task1/Dockerfile` هيكل عام غير مطابق بالضرورة لقالب GC الرسمي

## البناء التلقائي

عند أي push لمسار `task1/`، تعمل GitHub Action تلقائيًا لبناء الحاوية
ورفعها كـ Artifact قابل للتحميل (بدون الحاجة لـ Docker مثبت محليًا).
