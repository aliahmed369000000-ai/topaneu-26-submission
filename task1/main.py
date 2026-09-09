"""
نسخة متوافقة مع القالب الرسمي لـ TopAneu-26 Task 1.
المصدر: https://github.com/Bangulli/TopAneu-26/templates/task1/main.py
"""

import glob
import json
from pathlib import Path
from inference import infer_ct, infer_mr
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    interface_key = get_interface_key()

    handler = {
        ("head-ct-angiography",): interf0_handler,
        ("head-mr-angiography",): interf1_handler,
    }[interface_key]

    return handler()


def interf0_handler():
    input_head_ct_angiography = load_image_file(
        location=INPUT_PATH / "images/head-ct-angio",
    )

    output_detected_aneurysm_locations = infer_ct(input_head_ct_angiography)

    write_json_file(
        location=OUTPUT_PATH / "detected-aneurysm-locations.json",
        content=output_detected_aneurysm_locations,
    )

    return 0


def interf1_handler():
    input_head_mr_angiography = load_image_file(
        location=INPUT_PATH / "images/head-mr-angio",
    )

    output_detected_aneurysm_locations = infer_mr(input_head_mr_angiography)

    write_json_file(
        location=OUTPUT_PATH / "detected-aneurysm-locations.json",
        content=output_detected_aneurysm_locations,
    )

    return 0


def get_interface_key():
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file(*, location):
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])
    return result


if __name__ == "__main__":
    raise SystemExit(run())
