from typing import Literal
from fastapi import APIRouter, Query, HTTPException
from app.services.dataset_repository import (
    scan_motifs,
    scan_categories,
    scan_materials,
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
