#!/usr/bin/env python3
"""حوّل NIfTI إلى MHA + inputs.json لمحاكاة إدخال Grand Challenge."""
import argparse
import json
from pathlib import Path

import SimpleITK as sitk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="مسار .nii.gz أو .mha")
    p.add_argument("--modality", choices=["ct", "mr"], default="ct")
    p.add_argument("--out-dir", default="./test", help="مجلد test")
    args = p.parse_args()

    out = Path(args.out_dir)
    if args.modality == "ct":
        img_dir = out / "input" / "images" / "head-ct-angio"
        slug = "head-ct-angiography"
    else:
        img_dir = out / "input" / "images" / "head-mr-angio"
        slug = "head-mr-angiography"
    img_dir.mkdir(parents=True, exist_ok=True)
    (out / "output").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)

    src = Path(args.image)
    img = sitk.ReadImage(str(src))
    dst = img_dir / "image.mha"
    sitk.WriteImage(img, str(dst), useCompression=True)
    print(f"wrote {dst} size={img.GetSize()}")

    inputs = [
        {
            "file": str(dst.relative_to(out / "input")).replace("\\", "/"),
            "socket": {"slug": slug},
        }
    ]
    inputs_path = out / "input" / "inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2))
    print(f"wrote {inputs_path}")
    print("Next: MODEL_TAR=... ./do_test_run.sh   OR place weights in ./models/")


if __name__ == "__main__":
    main()
