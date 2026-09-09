"""
Download TopAneu training data from SwitchDrive via WebDAV (public share).
Resumable: skips files that already exist with matching size.
Only downloads what Task 1 needs: images, vessel_masks, location_jsons, location_mapping.json
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

SHARE = "O36U43RkChkNcHd"
BASE = "https://drive.switch.ch/public.php/webdav"
AUTH = HTTPBasicAuth(SHARE, "")
OUT = Path(os.environ.get("TOPANEU_OUT", "/kaggle/working/topaneu"))
# Only Task-1 essentials (skip location_masks, type_masks to save time/space)
NEEDED_DIRS = ["images", "vessel_masks", "location_jsons"]
NEEDED_FILES = ["location_mapping.json"]
MAX_WORKERS = 6


def propfind(path: str):
    url = BASE + path
    headers = {"Depth": "1", "Content-Type": "application/xml"}
    body = (
        '<?xml version="1.0"?>'
        '<d:propfind xmlns:d="DAV:">'
        "<d:prop><d:displayname/><d:getcontentlength/><d:resourcetype/></d:prop>"
        "</d:propfind>"
    )
    r = requests.request("PROPFIND", url, auth=AUTH, headers=headers, data=body, timeout=120)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"d": "DAV:"}
    items = []
    for resp in root.findall("d:response", ns):
        href = resp.find("d:href", ns).text
        prop = resp.find("d:propstat/d:prop", ns)
        if prop is None:
            continue
        name = prop.findtext("d:displayname", default="", namespaces=ns) or ""
        length = prop.findtext("d:getcontentlength", default="", namespaces=ns) or "0"
        is_coll = prop.find("d:resourcetype/d:collection", ns) is not None
        # normalize remote path relative to webdav root
        # href like /public.php/webdav/images/foo.nii.gz
        rel = href.split("/public.php/webdav", 1)[-1]
        if not rel.startswith("/"):
            rel = "/" + rel
        items.append({"rel": rel, "name": name, "size": int(length) if length.isdigit() else 0, "is_dir": is_coll})
    return items


def list_files(dir_path: str):
    """List files (not dirs) under dir_path like /images/"""
    if not dir_path.endswith("/"):
        dir_path += "/"
    items = propfind(dir_path)
    files = []
    for it in items:
        if it["is_dir"]:
            continue
        if it["rel"].rstrip("/") == dir_path.rstrip("/"):
            continue
        files.append(it)
    return files


def download_one(it, out_root: Path):
    rel = it["rel"]  # /images/foo.nii.gz
    local = out_root / rel.lstrip("/")
    local.parent.mkdir(parents=True, exist_ok=True)
    expected = it["size"]
    if local.is_file() and expected > 0 and local.stat().st_size == expected:
        return "skip", rel, expected
    if local.is_file() and expected > 0 and local.stat().st_size > 0 and local.stat().st_size != expected:
        local.unlink()  # incomplete
    url = BASE + rel
    with requests.get(url, auth=AUTH, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = local.with_suffix(local.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.rename(local)
    return "ok", rel, local.stat().st_size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT}")
    print("Listing remote files...")

    jobs = []
    for d in NEEDED_DIRS:
        files = list_files(f"/{d}/")
        print(f"  /{d}/ -> {len(files)} files")
        jobs.extend(files)
    for fname in NEEDED_FILES:
        # root-level file
        items = propfind("/")
        for it in items:
            if not it["is_dir"] and it["name"] == fname:
                jobs.append(it)
                break

    total_bytes = sum(j["size"] for j in jobs)
    print(f"Total to fetch: {len(jobs)} files, ~{total_bytes/1e9:.2f} GB")

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(download_one, j, OUT): j for j in jobs}
        with tqdm(total=len(jobs), unit="file") as bar:
            for fut in as_completed(futs):
                try:
                    status, rel, sz = fut.result()
                    if status == "skip":
                        skipped += 1
                    else:
                        done += 1
                except Exception as e:
                    failed += 1
                    j = futs[fut]
                    print(f"FAIL {j['rel']}: {e}")
                bar.update(1)
                bar.set_postfix(ok=done, skip=skipped, fail=failed)

    print(f"\nDone. ok={done} skipped={skipped} failed={failed}")
    # verify
    n_img = len(list((OUT / "images").glob("*.nii.gz"))) if (OUT / "images").is_dir() else 0
    n_ves = len(list((OUT / "vessel_masks").glob("*.nii.gz"))) if (OUT / "vessel_masks").is_dir() else 0
    n_json = len(list((OUT / "location_jsons").glob("*.json"))) if (OUT / "location_jsons").is_dir() else 0
    print(f"Local counts: images={n_img} vessel_masks={n_ves} location_jsons={n_json}")
    if n_img < 100:
        sys.exit(1)
    print("READY")


if __name__ == "__main__":
    main()
