from typing import Literal
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from urllib.parse import unquote
from app.services.dataset_repository import (
    scan_motifs,
    scan_categories,
    scan_materials,
    DATASET_DIR,
    validate_safe_path,
    IMAGE_EXTENSIONS,
)

router = APIRouter(tags=["dataset"])


@router.get("/motifs")
def get_motifs():
    """
    Returns all local traditional motif reference items scanned from dataset/Motif_*.
    """
    return scan_motifs()


@router.get("/categories")
def get_categories(
    type: Literal["makanan", "minuman"] = Query(
        "makanan",
        description="Category type: 'makanan' or 'minuman'"
    )
):
    """
    Returns product reference items scanned from dataset/Makanan_UMKM or dataset/Minuman_UMKM.
    """
    if type not in ("makanan", "minuman"):
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'type' must be 'makanan' or 'minuman'."
        )
    return scan_categories(type)


@router.get("/materials")
def get_materials():
    """
    Returns packaging material reference items scanned from dataset/Material_Kemasan.
    """
    return scan_materials()


@router.get("/dataset/image/{path:path}")
def serve_dataset_image(path: str):
    """
    Serves a dataset image file to Flutter frontend.
    Path is relative to the dataset root, e.g. 'Makanan_UMKM/Buras.jpeg'
    or 'Minuman_UMKM/Kopi Toraja.jpg' or 'Motif_Bugis/Lipa Sabbe.jpg'.

    Security: validates path is within dataset directory (no path traversal).
    """
    # Decode URL-encoded path (e.g. spaces, special chars)
    decoded_path = unquote(path).replace("+", " ")
    target = DATASET_DIR / decoded_path

    try:
        safe_path = validate_safe_path(target)
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Dataset image not found: {decoded_path}")

    # Determine media type from extension
    ext = safe_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    return FileResponse(
        path=str(safe_path),
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Access-Control-Allow-Origin": "*",
        }
    )
