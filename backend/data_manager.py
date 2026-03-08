"""
Kaggle Dataset Manager -- In-Memory Streaming
Fetches real knee X-ray images from Kaggle API without writing to disk.
Prevents data leakage: only test/val/auto_test splits are accessible.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO
import time

# Data leakage prevention
SAFE_SPLITS = {"test", "val", "auto_test"}
BLOCKED_SPLITS = {"train"}

DATASET_OWNER = "shashwatwork"
DATASET_SLUG = "knee-osteoarthritis-dataset-with-severity"
DATASET_VERSION = 1

KL_LABELS = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}

# In-memory caches -- zero disk I/O
_manifest_cache: Dict[str, list] = {}
_image_cache: Dict[str, bytes] = {}
_kaggle_client = None


def _get_client():
    global _kaggle_client
    if _kaggle_client is None:
        from kagglehub.clients import build_kaggle_client
        _kaggle_client = build_kaggle_client()
    return _kaggle_client


def _fetch_manifest(split: str, images_per_grade: int = 8) -> List[Dict]:
    """Page through the Kaggle file list and collect up to images_per_grade per KL grade."""
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest

    client = _get_client()
    collected: Dict[int, List[Dict]] = {g: [] for g in range(5)}
    page_token: Optional[str] = None

    while True:
        req = ApiListDatasetFilesRequest()
        req.owner_slug = DATASET_OWNER
        req.dataset_slug = DATASET_SLUG
        req.page_size = 200
        if page_token:
            req.page_token = page_token

        resp = client.datasets.dataset_api_client.list_dataset_files(req)
        batch = resp.files or []

        for f in batch:
            name: str = f.name or ""
            if not name.startswith(split + "/"):
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            try:
                grade = int(parts[1])
            except ValueError:
                continue
            if grade not in collected:
                continue
            if len(collected[grade]) < images_per_grade:
                collected[grade].append({
                    "filename": parts[2],
                    "kaggle_path": name,
                    "split": split,
                    "kl_grade": grade,
                    "kl_label": f"KL-{grade}: {KL_LABELS[grade]}",
                    "file_size": f.total_bytes or 0,
                })

        all_full = all(len(v) >= images_per_grade for v in collected.values())
        page_token = getattr(resp, "next_page_token", None)
        if all_full or not page_token or len(batch) == 0:
            break

    result = []
    for grade in range(5):
        result.extend(collected[grade])
    return result


def _stream_image_bytes(kaggle_path: str) -> bytes:
    """Download a single image from Kaggle into RAM (no disk write)."""
    if kaggle_path in _image_cache:
        return _image_cache[kaggle_path]

    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    client = _get_client()
    req = ApiDownloadDatasetRequest()
    req.owner_slug = DATASET_OWNER
    req.dataset_slug = DATASET_SLUG
    req.dataset_version_number = DATASET_VERSION
    req.file_name = kaggle_path

    response = client.datasets.dataset_api_client.download_dataset(req)
    if response.status_code != 200:
        raise RuntimeError(f"Kaggle {response.status_code} for {kaggle_path}")

    img_bytes = response.content

    # Bound cache to ~50 entries
    if len(_image_cache) >= 50:
        del _image_cache[next(iter(_image_cache))]
    _image_cache[kaggle_path] = img_bytes
    return img_bytes


#  Public API 

def list_available_images(split: str = "test", limit: Optional[int] = 50) -> List[Dict]:
    split = split.lower()
    if split in BLOCKED_SPLITS:
        print(f"[DataManager] Blocked access to '{split}' (data leakage prevention)")
        return []
    if split not in SAFE_SPLITS:
        return []

    if split not in _manifest_cache:
        print(f"[DataManager] Fetching manifest for split='{split}' from Kaggle...")
        t0 = time.time()
        _manifest_cache[split] = _fetch_manifest(split, images_per_grade=8)
        print(f"[DataManager] {len(_manifest_cache[split])} images cached ({time.time()-t0:.1f}s)")

    images = _manifest_cache[split]
    return images[:limit] if limit else images


def get_image_by_filename(filename: str, split: str = "test") -> Optional[Dict]:
    for img in list_available_images(split, limit=None):
        if img["filename"] == filename:
            return img
    return None


def load_image_bytes(img_dict: Dict) -> bytes:
    return _stream_image_bytes(img_dict["kaggle_path"])


def get_statistics(split: str = "test") -> Dict:
    split = split.lower()
    safe = split not in BLOCKED_SPLITS and split in SAFE_SPLITS
    images = list_available_images(split) if safe else []
    grade_counts = {i: sum(1 for img in images if img["kl_grade"] == i) for i in range(5)}
    total_bytes = sum(img["file_size"] for img in images)
    return {
        "split": split,
        "total_images": len(images),
        "grade_distribution": grade_counts,
        "total_size_kb": round(total_bytes / 1024, 1),
        "safe_to_use": safe,
    }


def setup_credentials(credentials_path: Path):
    import shutil, os
    dest = Path.home() / ".kaggle" / "kaggle.json"
    if not dest.exists() and credentials_path.exists():
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(credentials_path, dest)
        try: os.chmod(dest, 0o600)
        except Exception: pass
        print(f"[DataManager] Credentials installed at {dest}")
