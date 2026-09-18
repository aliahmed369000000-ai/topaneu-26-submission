"""
معمارية Task 2: تجزئة ثلاثية الأبعاد متعددة الفئات (53 فئة: خلفية + 52
موقعًا تشريحيًا). راجع docs/CHALLENGE_REFERENCE_TASK2.md للحقائق الكاملة.

الفرق عن task1/model.py: هذه Encoder-Decoder كاملة (U-Net) بوصلات تخطي
بين كل مستوى دقة مكانية، تُخرج قناعًا بنفس أبعاد الإدخال المكانية —
لا ضغط لمتجه واحد كما في Task 1 (يفقد الموقع المكاني، غير مناسب هنا).
"""

import torch
import torch.nn as nn

N_CLASSES_TASK2 = 53  # 0=خلفية + 1..52=مواقع (نفس جدول Task 1)


class ConvBlock3D(nn.Module):
    """طبقتا Conv3D متتاليتان + BatchNorm + ReLU، بدون تغيير الأبعاد المكانية."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """U-Net ثلاثي الأبعاد قياسي: 4 مراحل تصغير (Encoder) + 4 مراحل
    تكبير (Decoder) بوصلات تخطي (Concatenation) عند كل مستوى.

    channels: قنوات كل مرحلة في الـEncoder (تتضاعف تقريبًا كل مرحلة،
    بنفس روح Backbone3D_v1 في Task 1 لكن مع الاحتفاظ بخرائط كل مستوى
    للـDecoder بدل GAP في النهاية)."""

    def __init__(self, in_channels: int = 2, n_classes: int = N_CLASSES_TASK2,
                 channels: tuple = (16, 32, 64, 128, 256)):
        super().__init__()
        self._n_downsample_stages = 4  # يحدد قيد القسمة أدناه؛ حدّثه لو غيّرت عدد المراحل
        c1, c2, c3, c4, c5 = channels

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, c1)
        self.enc2 = ConvBlock3D(c1, c2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.enc4 = ConvBlock3D(c3, c4)
        self.bottleneck = ConvBlock3D(c4, c5)
        self.pool = nn.MaxPool3d(2)

        # Decoder (ConvTranspose3d للتكبير + دمج وصلة التخطي عبر Concatenation)
        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(c4 + c4, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c1 + c1, c1)

        # طبقة تصنيف نهائية: 1x1x1 conv لكل فوكسل -> n_classes (logits خام)
        self.out_conv = nn.Conv3d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        divisor = 2 ** self._n_downsample_stages
        for dim_name, dim_size in zip(("D", "H", "W"), x.shape[2:]):
            assert dim_size % divisor == 0, (
                f"بُعد الإدخال {dim_name}={dim_size} يجب أن يقبل القسمة على {divisor} "
                f"(= 2^{self._n_downsample_stages} مراحل تصغير) وإلا لن تتطابق أبعاد "
                f"وصلات التخطي (Skip Connections) بين Encoder وDecoder. مثال صحيح: 128 "
                f"(الحجم الحقيقي المستهدف). مثال خاطئ: 24 (اكتُشف فعليًا أثناء الاختبار)."
            )

        # Encoder (نحفظ كل مستوى قبل التصغير لوصلات التخطي)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # (batch, n_classes, D, H, W) logits خام


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = UNet3D()
    # حجم صغير للتحقق البنيوي فقط (128^3 الحقيقي يحتاج ذاكرة أكبر بكثير
    # من بيئة الاختبار هذه؛ الشكل النسبي والمنطق يبقيان صحيحين لأي حجم
    # يقبل القسمة على 16 = 2^4 بسبب 4 مراحل تصغير)
    dummy = torch.randn(1, 2, 32, 32, 32)
    net.eval()
    with torch.no_grad():
        out = net(dummy)
    assert out.shape == (1, N_CLASSES_TASK2, 32, 32, 32), f"شكل غير متوقع: {out.shape}"

    n_params = count_parameters(net)
    print(f"شكل الإخراج: {tuple(out.shape)} (متوقع: (1, {N_CLASSES_TASK2}, 32, 32, 32)) ✓")
    print(f"إجمالي المعاملات: {n_params:,} ({n_params * 4 / 1e6:.1f} MB, fp32)")
