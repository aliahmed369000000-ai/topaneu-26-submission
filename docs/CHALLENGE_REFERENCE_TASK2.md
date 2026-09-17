# مرجع التحدي - TopAneu 2026 (Task 2)

هذا الملف يكمّل `CHALLENGE_REFERENCE.md` (الخاص بـTask 1). راجعه أولًا
للحقائق المشتركة (قيود المنصة، جدول الفئات الـ52، flip_map، بيانات
التدريب). هنا فقط ما يخص Task 2 تحديدًا.

## تعريف المهمة (مؤكد من الصفحة الرئيسية الرسمية)

"Given a scan, detect and segment aneurysms at the voxel level,
assigning each aneurysm to its vessel location class."

**فرق جوهري عن Task 1:** Task 1 تصنيف عام للفحص (52 رقمًا). Task 2
**تجزئة كثيفة (Dense Segmentation)** — كل فوكسل في الحجم الثلاثي
الأبعاد يحصل على تصنيفه الخاص. يتطلب معمارية Encoder-Decoder (U-Net)،
لا يمكن إعادة استخدام Backbone3D_v1 + رأس Dense من Task 1 كما هو
(يضغط كل شيء لمتجه واحد، يفقد الموقع المكاني).

## بيانات Task 2 (مؤكدة من /data/ الرسمية، نفس الحقائق مصدرها Task 1)

- `location_masks/`: **قناع التجزئة الأساسي (Ground Truth)** — قناع
  واحد لكل حالة، بنفس أبعاد `images/`. كل فوكسل تمدد له قيمة = فئة
  موقعه (1-52، **نفس جدول location_mapping.json المستخدم في Task 1
  بالضبط**). الباقي (لا تمدد) = 0 (خلفية).
  → **53 فئة إخراج فعليًا: 0=خلفية + 1..52=المواقع.**
- `type_masks/`: قناع نوع التمدد (saccular/dissecting/fusiform، 3
  فئات، `type_mapping.json`). **ليس إخراجًا مطلوبًا** — بيانات مساعدة
  اختيارية فقط (يمكن استخدامها كإشارة تدريب إضافية، بنفس منطق
  استخدامنا لـ`vessel_masks` في Task 1).
- `vessel_masks/`: نفس ملف Task 1 بالضبط (silver-standard، من نموذج
  TopBrain) — نفس الدور المساعد (قناة إدخال إضافية محتملة).

## قيود المنصة والاستدلال (لم تُتأكد بشكل مستقل لـTask 2، الأرجح مطابقة لـTask 1)

- نفس بنية GC التحتية (T4 16GB، RAM 32GB، ≤12 دقيقة/حالة) — لم يُؤكَّد
  رقم مستقل لـTask 2 من /participation/، لكن هذا نفس النظام الأساسي.
- توجد مرحلة "فحص أولي" (`Task 2: preliminary docker evaluation`)
  ومرحلة نهائية (`Task 2: final test phase`) — نفس نمط Task 1، راجع
  صفحة `/challenge/statistics/` للأرقام الحالية عند الحاجة.
- ⚠️ لم نؤكد بعد: صيغة `infer_ct`/`infer_mr` الدقيقة المطلوبة لـTask 2
  (هل تُعيد قناع segmentation كملف NIfTI، أم صيغة أخرى؟) — يحتاج تحقق
  من main.py الرسمي لـTask 2 قبل بناء `inference.py` النهائي (نفس ما
  فعلناه لـTask 1 عبر Bangulli/TopAneu-26).

## نقطة مفتوحة

- مقياس التقييم الرسمي لـTask 2 لم يُؤكَّد بشكل مستقل (الأرجح Dice-based
  بناءً على نمط كل تحديات segmentation المشابهة على GC، لكن هذا **تخمين
  مبني على سوابق، لا حقيقة مؤكدة من صفحة evaluation الرسمية لهذا
  التحدي تحديدًا** — يستحق تحقق مباشر لاحقًا).
