"""
معالجة حالة واحدة: تحميل الصورة + قناع الأوعية، Resize موحّد إلى 128^3،
مع احترام الفرق الجوهري بين نوعي البيانات:
  - الصورة (intensity): قيم مستمرة -> استيفاء خطي (order=1)
  - قناع الأوعية (categorical): قيم تصنيفية -> Nearest-Neighbor حصرًا (order=0)
    ثم تحويله لقناع ثنائي (binary) بدل القيم الرقمية الخام، كما هو موثّق
    في نقطة القرار المفتوحة سابقًا.

الإخراج: مصفوفة (2, 128, 128, 128) بترتيب القنوات:
  channel 0 = intensity (بعد z-score)
  channel 1 = vessel mask (binary: 0/1)
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def load_and_resize_case(
    image_path: str,
    vessel_mask_path: str,
    target_shape: tuple = (128, 128, 128),
) -> np.ndarray:
    img_nii = nib.load(image_path)
    image = img_nii.get_fdata().astype(np.float32)

    vessel_nii = nib.load(vessel_mask_path)
    vessel = vessel_nii.get_fdata()

    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]

    # الصورة: استيفاء خطي
    image_resized = zoom(image, zoom_factors, order=1)

    # قناع الأوعية: Nearest-Neighbor حصرًا (بيانات تصنيفية) — لا يجوز order=1/3 هنا
    vessel_resized = zoom(vessel, zoom_factors, order=0)

    # تحويل لقناع ثنائي
    vessel_binary = (vessel_resized > 0).astype(np.float32)

    # تطبيع الشدة (z-score) على الفوكسلات غير الصفرية فقط لتفادي هيمنة الخلفية السوداء
    nonzero = image_resized[image_resized > 0]
    if nonzero.size > 0:
        mean, std = nonzero.mean(), nonzero.std() + 1e-8
        image_resized = (image_resized - mean) / std

    volume = np.stack([image_resized, vessel_binary], axis=0)
    assert volume.shape == (2, *target_shape), f"شكل غير متوقع: {volume.shape}"
    return volume.astype(np.float32)


# مصدر وحيد لخريطة تبديل التسميات عند القلب يمين/يسار (Flip Augmentation).
# 0-based (يطابق ترقيم الفئات 0..51 في مصفوفة Y مباشرة، بدون أي طرح/إضافة).
# مأخوذة من جدول location_mapping.json الرسمي: 48 من 52 فئة مُقسّمة أزواج
# R/L متطابقة الاسم؛ 6, 7, 16, 35 (BA trunk, VA-BA junction, BA tip,
# Acom complex) بدون جانبية (NA) فتبقى كما هي (غير موجودة في القاموس أدناه،
# استخدم .get(i, i) عند التطبيق).
FLIP_MAP = {
    0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 8: 9, 9: 8, 10: 11, 11: 10,
    12: 13, 13: 12, 14: 15, 15: 14, 17: 18, 18: 17, 19: 20, 20: 19,
    21: 22, 22: 21, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
    29: 30, 30: 29, 31: 32, 32: 31, 33: 34, 34: 33, 36: 37, 37: 36,
    38: 39, 39: 38, 40: 41, 41: 40, 42: 43, 43: 42, 44: 45, 45: 44,
    46: 47, 47: 46, 48: 49, 49: 48, 50: 51, 51: 50,
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python vessel_preprocessing.py <image.nii.gz> <vessel_mask.nii.gz>")
        sys.exit(1)

    vol = load_and_resize_case(sys.argv[1], sys.argv[2])
    print("shape:", vol.shape)
    print("intensity channel range:", vol[0].min(), vol[0].max())
    print("vessel channel unique values:", np.unique(vol[1]))
