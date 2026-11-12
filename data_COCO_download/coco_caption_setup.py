import argparse
import hashlib
import random
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_with_resume(
    url: str,
    dest: Path,
    *,
    retries: int = 20,
    backoff_base: float = 1.7,
    timeout_sec: int = 60,
    chunk_size: int = 1024 * 1024,
    user_agent: str = "coco-downloader/1.0",
):
    """
    Robust downloader:
    - streams in chunks
    - resumes via HTTP Range if partial file exists
    - retries with exponential backoff + jitter
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    # If file exists, try resume
    existing = dest.stat().st_size if dest.exists() else 0

    # Get remote size (if available)
    remote_size = None
    try:
        head = session.head(url, timeout=timeout_sec, allow_redirects=True)
        if head.ok and head.headers.get("Content-Length"):
            remote_size = int(head.headers["Content-Length"])
    except requests.RequestException:
        pass

    if remote_size is not None and existing == remote_size:
        print(f"[SKIP] Already downloaded: {dest.name} ({existing} bytes)")
        return

    for attempt in range(retries):
        try:
            headers = {}
            mode = "wb"
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                mode = "ab"

            with session.get(url, stream=True, timeout=timeout_sec, headers=headers, allow_redirects=True) as r:
                r.raise_for_status()

                # Determine total for progress bar
                total = None
                if "Content-Range" in r.headers:
                    # e.g. "bytes 123-999/1000"
                    total = int(r.headers["Content-Range"].split("/")[-1])
                elif "Content-Length" in r.headers and remote_size is None:
                    # length of response (might be partial)
                    resp_len = int(r.headers["Content-Length"])
                    total = existing + resp_len
                else:
                    total = remote_size

                pbar = tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dest.name}",
                )

                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            existing += len(chunk)
                            pbar.update(len(chunk))

                pbar.close()

            # Final size check if we know remote size
            if remote_size is not None:
                final = dest.stat().st_size
                if final != remote_size:
                    raise RuntimeError(f"Size mismatch: got {final}, expected {remote_size}")

            print(f"[OK] Downloaded: {dest} ({dest.stat().st_size} bytes)")
            return

        except Exception as e:
            wait = (backoff_base ** attempt) + random.random()
            print(f"[WARN] download failed (attempt {attempt+1}/{retries}): {e}")
            print(f"       retrying in {wait:.1f}s ...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download after {retries} retries: {url}")


def unzip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("[OK] Extracted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="coco_data", help="dataset root folder")
    ap.add_argument("--download-images", action="store_true", help="also download train/val images (large)")
    args = ap.parse_args()

    root = Path(args.root)
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    # COCO 2017
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    train_img_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_img_url = "http://images.cocodataset.org/zips/val2017.zip"

    ann_zip = downloads / "annotations_trainval2017.zip"
    download_with_resume(ann_url, ann_zip)

    # Extract annotations
    unzip(ann_zip, root)

    if args.download_images:
        train_zip = downloads / "train2017.zip"
        val_zip = downloads / "val2017.zip"

        download_with_resume(train_img_url, train_zip)
        download_with_resume(val_img_url, val_zip)

        unzip(train_zip, root)
        unzip(val_zip, root)

    print("\n[DONE] Folder layout example:")
    print(f"  {root}/annotations/captions_train2017.json")
    print(f"  {root}/annotations/captions_val2017.json")
    if args.download_images:
        print(f"  {root}/train2017/*.jpg")
        print(f"  {root}/val2017/*.jpg")


if __name__ == "__main__":
    main()
