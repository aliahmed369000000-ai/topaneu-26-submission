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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python vessel_preprocessing.py <image.nii.gz> <vessel_mask.nii.gz>")
        sys.exit(1)

    vol = load_and_resize_case(sys.argv[1], sys.argv[2])
    print("shape:", vol.shape)
    print("intensity channel range:", vol[0].min(), vol[0].max())
    print("vessel channel unique values:", np.unique(vol[1]))
