import os
from pathlib import Path
from typing import Any
from fastapi import HTTPException

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_dataset_dir() -> Path:
    """
    Locates the dataset directory on the filesystem.
    Supports environment variable DATASET_DIR, or falls back to common candidate locations.
    Supports HuggingFace Spaces (working dir /home/user/app/) and local dev environments.
    """
    env_path = os.getenv("DATASET_DIR")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p

    base_backend_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    cwd = Path.cwd()

    candidates = [
        base_backend_dir / "assets" / "dataset",              # backend/assets/dataset (lokal)
        base_backend_dir.parent / "assets" / "dataset",       # app_desainku/assets/dataset (lokal)
        base_backend_dir / "dataset",                         # backend/dataset
        # HuggingFace Space: app.py di /home/user/app/, file ini di /home/user/app/app/services/
        Path(__file__).resolve().parent.parent.parent / "assets" / "dataset",
        Path("/home/user/app/assets/dataset"),                 # HuggingFace Spaces (Python SDK)
        Path("/home/user/app/app/assets/dataset"),             # HuggingFace alt structure
        cwd / "assets" / "dataset",                           # Relative to CWD
        cwd / "dataset",                                      # Relative to CWD
        Path("assets/dataset").resolve(),
        Path("dataset").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    # Default fallback — log warning
    fallback = (base_backend_dir / "assets" / "dataset").resolve()
    return fallback


DATASET_DIR = get_dataset_dir()

# In-memory cache for scanned dataset items
_dataset_cache: dict[str, list[dict[str, Any]]] = {}


def clear_dataset_cache():
    """
    Invalidates the in-memory scan cache.
    """
    global _dataset_cache
    _dataset_cache.clear()


def validate_safe_path(target_path: Path) -> Path:
    """
    Prevents path traversal attacks by ensuring the target path is strictly within
    the dataset root directory and points to an existing file.
    """
    dataset_resolved = DATASET_DIR.resolve()
    target_resolved = target_path.resolve()

    try:
        target_resolved.relative_to(dataset_resolved)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Access denied: Invalid path traversal attempt."
        )

    if not target_resolved.exists() or not target_resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found in dataset: {target_path.name}"
        )

    return target_resolved


def scan_dataset_folder(folder_root: Path) -> list[Path]:
    """
    Generic scanner function that recursively collects all supported image files
    under a given root directory.
    """
    if not folder_root.exists() or not folder_root.is_dir():
        return []

    image_files = []
    for file_path in folder_root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)

    image_files.sort(key=lambda p: str(p).lower())
    return image_files


def _clean_name(name: str) -> str:
    """
    Clean filename or folder name into a user-friendly label.
    Strips leading dots/spaces, replaces underscores with spaces.
    """
    cleaned = name.strip()
    while cleaned.startswith(".") or cleaned.startswith(" "):
        cleaned = cleaned[1:].strip()
    return cleaned.replace("_", " ")


def scan_motifs() -> list[dict[str, Any]]:
    """
    Scans dataset/Motif_* directories for local motifs.
    Response format:
    [
      {
        "region": "Makassar",
        "kabupaten": "Gowa",
        "motif": "Butta Toa",
        "filename": "Butta_Toa.png",
        "path": "Motif_Makassar/Motif_Gowa/Butta_Toa.png"
      }
    ]
    """
    if "motifs" in _dataset_cache:
        return _dataset_cache["motifs"]

    motifs_list: list[dict[str, Any]] = []
    if not DATASET_DIR.exists():
        return motifs_list

    # Search for all subfolders starting with Motif_
    motif_root_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith("Motif_")]
    motif_root_dirs.sort(key=lambda d: d.name)

    for motif_dir in motif_root_dirs:
        # Extract region name (e.g. "Motif_Bugis" -> "Bugis")
        region_raw = motif_dir.name.replace("Motif_", "").strip()

        image_files = scan_dataset_folder(motif_dir)
        for img_path in image_files:
            rel_path = img_path.relative_to(DATASET_DIR)
            parts = rel_path.parts  # e.g., ('Motif_Bugis', 'Motif Barru', 'Motif Lontara.jpg') or ('Motif_Toraja', 'Pa\' Tedong.jpg')

            if len(parts) >= 3:
                # Subfolder exists (e.g., 'Motif Barru')
                kab_raw = parts[1]
                kabupaten = kab_raw.replace("Motif ", "").replace("Motif_", "").strip()
            else:
                kabupaten = region_raw

            filename = img_path.name
            base_name = img_path.stem
            clean_motif = _clean_name(base_name)
            if clean_motif.lower().startswith("motif "):
                motif_name = clean_motif[6:].strip()
            else:
                motif_name = clean_motif

            motifs_list.append({
                "region": region_raw,
                "kabupaten": kabupaten,
                "motif": motif_name,
                "filename": filename,
                "path": rel_path.as_posix()
            })

    _dataset_cache["motifs"] = motifs_list
    return motifs_list


def scan_categories(category_type: str) -> list[dict[str, Any]]:
    """
    Scans dataset/Makanan_UMKM or dataset/Minuman_UMKM.
    Response format:
    [
      {
        "type": "makanan",
        "sub_category": "Abon Ikan",
        "filename": "Abon Ikan.jpeg",
        "path": "Makanan_UMKM/Abon Ikan.jpeg"
      }
    ]
    """
    cat_type_clean = category_type.lower().strip()
    cache_key = f"category_{cat_type_clean}"
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    folder_name = "Makanan_UMKM" if cat_type_clean == "makanan" else "Minuman_UMKM"
    cat_dir = DATASET_DIR / folder_name

    categories_list: list[dict[str, Any]] = []
    if not cat_dir.exists():
        return categories_list

    image_files = scan_dataset_folder(cat_dir)
    for img_path in image_files:
        rel_path = img_path.relative_to(DATASET_DIR)
        parts = rel_path.parts  # e.g., ('Makanan_UMKM', 'Abon Ikan.jpeg') or ('Makanan_UMKM', 'Olahan_Ikan', 'Abon.jpg')

        if len(parts) > 2:
            sub_cat = _clean_name(parts[1])
        else:
            sub_cat = _clean_name(img_path.stem)

        categories_list.append({
            "type": cat_type_clean,
            "sub_category": sub_cat,
            "filename": img_path.name,
            "path": rel_path.as_posix()
        })

    _dataset_cache[cache_key] = categories_list
    return categories_list


def scan_materials() -> list[dict[str, Any]]:
    """
    Scans dataset/Material_Kemasan directory.
    Response format:
    [
      {
        "material": "Kertas Kraft",
        "filename": "Kertas Kraft.jpg",
        "path": "Material_Kemasan/Kertas Kraft.jpg"
      }
    ]
    """
    if "materials" in _dataset_cache:
        return _dataset_cache["materials"]

    mat_dir = DATASET_DIR / "Material_Kemasan"
    materials_list: list[dict[str, Any]] = []
    if not mat_dir.exists():
        return materials_list

    image_files = scan_dataset_folder(mat_dir)
    for img_path in image_files:
        rel_path = img_path.relative_to(DATASET_DIR)
        parts = rel_path.parts  # e.g., ('Material_Kemasan', 'Kertas Kraft.jpg') or ('Material_Kemasan', 'Kraft_Paper', 'sample.jpg')

        if len(parts) > 2:
            material_name = _clean_name(parts[1])
        else:
            material_name = _clean_name(img_path.stem)

        materials_list.append({
            "material": material_name,
            "filename": img_path.name,
            "path": rel_path.as_posix()
        })

    _dataset_cache["materials"] = materials_list
    return materials_list


def resolve_motif_path(region: str, kabupaten: str, motif_filename: str) -> Path:
    """
    Resolves a motif image file path from region, kabupaten, and motif_filename.
    Validates file existence and prevents path traversal (returns 404 if not found or unsafe).
    """
    all_motifs = scan_motifs()

    reg_clean = region.lower().strip()
    kab_clean = kabupaten.lower().strip()
    file_clean = motif_filename.lower().strip()

    # 1. Exact / Fuzzy match search in scanned motifs
    for m in all_motifs:
        m_reg = m["region"].lower()
        m_kab = m["kabupaten"].lower()
        m_file = m["filename"].lower()
        m_name = m["motif"].lower()

        if m_reg == reg_clean and m_kab == kab_clean:
            if m_file == file_clean or m_name == file_clean:
                target_path = DATASET_DIR / m["path"]
                return validate_safe_path(target_path)

    # 2. Direct path construction fallback
    candidates = [
        DATASET_DIR / f"Motif_{region}" / f"Motif {kabupaten}" / motif_filename,
        DATASET_DIR / f"Motif_{region}" / f"Motif_{kabupaten}" / motif_filename,
        DATASET_DIR / f"Motif_{region}" / kabupaten / motif_filename,
        DATASET_DIR / f"Motif_{region}" / motif_filename,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return validate_safe_path(candidate)

    raise HTTPException(
        status_code=404,
        detail=f"Motif file '{motif_filename}' not found for region '{region}' and kabupaten '{kabupaten}'."
    )


def resolve_category_path(
    category_type: str,
    subfolder: str | None = None,
    filename: str | None = None
) -> Path:
    """
    Resolves a category image file path from category_type, subfolder, and optional filename.
    If filename is None, selects a representative image matching subfolder.
    """
    all_cats = scan_categories(category_type)

    if not all_cats:
        raise HTTPException(
            status_code=404,
            detail=f"No dataset images found for category type '{category_type}'."
        )

    file_clean = filename.lower().strip() if filename else None
    sub_clean = subfolder.lower().strip() if subfolder else None

    # Search for matching item
    for c in all_cats:
        c_file = c["filename"].lower()
        c_sub = c["sub_category"].lower()

        if file_clean and c_file == file_clean:
            return validate_safe_path(DATASET_DIR / c["path"])

        if sub_clean and c_sub == sub_clean and not file_clean:
            return validate_safe_path(DATASET_DIR / c["path"])

    # Fallback if only subfolder matched loosely or fallback to first item
    if sub_clean:
        for c in all_cats:
            if sub_clean in c["sub_category"].lower() or sub_clean in c["filename"].lower():
                return validate_safe_path(DATASET_DIR / c["path"])

    # First item fallback
    first_path = DATASET_DIR / all_cats[0]["path"]
    return validate_safe_path(first_path)


def resolve_material_path(material: str) -> Path:
    """
    Resolves a material texture/shape image from Material_Kemasan directory.
    """
    all_mats = scan_materials()
    mat_clean = material.lower().strip()

    for m in all_mats:
        if m["material"].lower() == mat_clean or m["filename"].lower() == mat_clean:
            return validate_safe_path(DATASET_DIR / m["path"])

    # Loose match fallback
    for m in all_mats:
        if mat_clean in m["material"].lower() or mat_clean in m["filename"].lower():
            return validate_safe_path(DATASET_DIR / m["path"])

    # If no image found but material name requested, check if folder exists or raise 404
    raise HTTPException(
        status_code=404,
        detail=f"Material '{material}' not found in Material_Kemasan dataset."
    )
