"""
هيكل لملف templates/task1/inference.py من مستودع github.com/Bangulli/TopAneu-26

=== ما هو مؤكد فعليًا من README الرسمي للمستودع (وليس تخمينًا) ===
- القالب فيه ملفان: main.py (توفره GC، يقرأ الصور ويغذّيها كـ Numpy Array)
  و inference.py (هنا فقط تضع كودك الخاص).
- الإخراج المتوقع: JSON يحتوي قائمة القيم الصحيحة (Integer) للفئات
  الموجبة فقط — المثال الرسمي في json_schema.json هو: [42, 13]
  (أي ليس متجه sigmoid كامل بطول 52، بل قائمة الفئات المُتوقَّع وجودها فقط).
- مسار الإخراج في الاختبار المحلي:
  ./eval/task1/test/input/[UID]/output/predicted-aneurysm-location.json
- قيود المنصة: 12 دقيقة/حالة، 32GB RAM، NVIDIA T4 16GB VRAM.

=== ⚠️ غير مؤكد — يتطلب تحقق يدوي منك بعد تحميل المستودع فعليًا ===
لم أستطع جلب محتوى main.py الفعلي (GitHub يحجب الوصول الآلي لعرض الشجرة/الملفات
الخام لهذا المسار). لذلك التوقيع أدناه (اسم الدالة، أسماء المعاملات، مفتاح
الـJSON "predicted_locations") هو الأنسب منطقيًا بناءً على ما هو موثّق، لكنه
غير مؤكد حرفيًا. قبل التعويل عليه:
  1. نزّل المستودع: git clone https://github.com/Bangulli/TopAneu-26
  2. افتح templates/task1/main.py وابحث عن الدالة التي يستدعيها من inference.py
     (غالبًا اسمها شيء مثل run() أو predict() أو infer())
  3. افتح eval/task1/test_evaluations/test.py ودالة get_predictions_entry
     للتأكد من اسم المفتاح المتوقع بالضبط في ملف JSON المُخرَج
  4. عدّل التوقيع والمفتاح أدناه ليطابق ما وجدته بالضبط
"""

import numpy as np
import torch

from vessel_preprocessing import load_and_resize_case  # نفس ملف المعالجة أعلاه

MODEL_WEIGHTS_PATH = "./models/topaneu_task1_final.pt"
DECISION_THRESHOLD = 0.5  # يمكن ضبطه لكل فئة على حدة بعد معايرة على مجموعة Validation

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def _load_model_once():
    global _MODEL
    if _MODEL is None:
        _MODEL = torch.load(MODEL_WEIGHTS_PATH, map_location=_DEVICE)
        _MODEL.eval()
    return _MODEL


def predict(image: np.ndarray, vessel_mask: np.ndarray) -> dict:
    """
    ⚠️ التوقيع (اسم الدالة + المعاملات) يجب مطابقته لما يستدعيه main.py فعليًا.

    image, vessel_mask: مصفوفات Numpy بالشكل الأصلي كما تُسلَّم من main.py
    (شكلها الأصلي غير موحّد بين الحالات — المعالجة أدناه توحّدها لـ128^3).
    """
    model = _load_model_once()

    volume = load_and_resize_case_from_arrays(image, vessel_mask)  # (2,128,128,128)
    tensor = torch.from_numpy(volume).unsqueeze(0).to(_DEVICE)  # (1,2,128,128,128)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()  # (52,)

    # تحويل من index (0-based في متجه الشبكة) إلى value (1-based كما في
    # location_mapping.json) — الفهرسة مؤكدة من الملف الرسمي المرفق سابقًا
    predicted_values = [int(idx + 1) for idx, p in enumerate(probs) if p >= DECISION_THRESHOLD]

    return {"predicted_locations": predicted_values}  # ⚠️ اسم المفتاح غير مؤكد — تحقق منه


def load_and_resize_case_from_arrays(image: np.ndarray, vessel_mask: np.ndarray,
                                      target_shape=(128, 128, 128)) -> np.ndarray:
    """نفس منطق vessel_preprocessing.load_and_resize_case لكن من مصفوفات جاهزة
    بدل مسارات ملفات (لأن main.py في القالب الرسمي يسلّم Numpy Array مباشرة)."""
    from scipy.ndimage import zoom

    image = image.astype(np.float32)
    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]

    image_resized = zoom(image, zoom_factors, order=1)
    vessel_resized = zoom(vessel_mask, zoom_factors, order=0)  # NN حصرًا
    vessel_binary = (vessel_resized > 0).astype(np.float32)

    nonzero = image_resized[image_resized > 0]
    if nonzero.size > 0:
        mean, std = nonzero.mean(), nonzero.std() + 1e-8
        image_resized = (image_resized - mean) / std

    return np.stack([image_resized, vessel_binary], axis=0).astype(np.float32)
