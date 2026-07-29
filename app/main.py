import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.routers import dataset
from app.services.dataset_repository import resolve_material_path
from app.services.stability_client import (
    resolve_init_image,
    generate_stability_image,
    resize_to_allowed_dimension,
    validate_api_key_on_startup,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enforce security validation at startup
    validate_api_key_on_startup()
    yield


app = FastAPI(
    title="SulselPak AI Backend API",
    description="FastAPI Service for UMKM Packaging Design Generation using Stability AI & Local Motif Dataset",
    version="3.0.0",
    lifespan=lifespan,
)

# Enable CORS for Flutter web/desktop/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Format HTTP exceptions as clear JSON responses compatible with Flutter frontend.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": exc.detail,
        },
    )


# Include dataset router
app.include_router(dataset.router)


class DesignRequest(BaseModel):
    description: str

    category_type: str = "makanan"                 # "makanan" | "minuman"
    category_subfolder: str | None = None          # contoh: nama sub-folder atau nama item
    category_filename: str | None = None           # opsional: nama file spesifik

    region: str = "Bugis"                          # "Bugis" | "Makassar" | "Toraja"
    kabupaten: str = "Barru"                       # contoh: "Gowa"
    motif_filename: str = "Lipa Sabbe.png"         # contoh: "Butta_Toa.png"

    material: str | None = None                    # nama sub-folder/material di Material_Kemasan
    eco: bool | None = None                        # field lama (backward compatibility)

    image: str | None = None                       # upload custom dari user (base64)
    use_motif_as_reference: bool = False           # jika True, pakai gambar motif sbg init_image
    use_category_as_reference: bool = False        # jika True, pakai gambar kategori sbg init_image


def _get_packaging_type(category_type: str, material: str) -> tuple[str, str]:
    """
    Returns (packaging_type_desc, packaging_shape) based on category and material.
    """
    cat = category_type.lower()
    mat = material.lower()

    if "botol" in mat or "kaca" in mat or "minuman" in cat or "drink" in cat or "beverage" in cat:
        return (
            "hyperrealistic 3D commercial glass jar container mockup with shiny metallic gold screw lid",
            "transparent glass jar with realistic reflections showing fresh food product contents inside, wrapped front label sticker"
        )
    elif "pouch" in mat or "plastik" in mat:
        return (
            "hyperrealistic 3D commercial stand-up zip pouch food packaging bag",
            "sealed matte finish zip-lock pouch bag with transparent viewing window and crisp front label printing"
        )
    elif "lontar" in mat or "pisang" in mat or "pelepah" in mat or "anyaman" in mat:
        return (
            "hyperrealistic 3D artisan handwoven eco packaging container",
            f"artisan woven {material} container with woven lid and custom printed label tag sleeve"
        )
    elif "tenun" in mat or "kain" in mat:
        return (
            "hyperrealistic 3D fabric-wrapped gift packaging box",
            f"elegant {material} wrapped gift box with printed ethnic label sleeve"
        )
    else:
        return (
            "hyperrealistic 3D kraft paper cardboard food packaging box",
            f"rectangular {material} food box with printed front label panel, crisp edges, studio lighting"
        )


def build_prompt(
    category_type: str,
    category_label: str,
    description: str,
    motif_name: str,
    kabupaten: str,
    region: str,
    material: str,
    product_name: str = "",
    brand_name: str = "",
    color_hint: str = "",
    is_halal: bool = False,
    target_market: str = "Lokal",
) -> str:
    """
    Build a detailed 3D packaging product mockup prompt for FLUX and Stability AI.
    """
    display_name = brand_name or product_name or category_label
    pack_type, pack_shape = _get_packaging_type(category_type, material)

    # Color guidance
    color_guide = ""
    if color_hint:
        color_guide = f"Color palette: {color_hint}. "
    
    # Certification badges
    badges = ""
    if is_halal:
        badges += "Halal certified logo badge visible on label. "

    # Market style
    market_style = (
        "sleek modern commercial retail product packaging, clean minimal layout"
        if target_market.lower() in ("nasional", "national")
        else "authentic artisan local market product packaging, warm earthy tones"
    )

    prompt = (
        f"Award-winning 3D commercial product packaging mockup of {pack_type}. "
        f"PACKAGING SHAPE: {pack_shape}. "
        f"PHYSICAL FRONT LABEL: A crisp, solid white rectangular label sticker physically attached directly onto the front surface of the packaging. "
        f"PRODUCT TITLE ON PACKAGING: Large, ultra-sharp, bold dark typography titled '{display_name.upper()}' printed prominently in the center of the front packaging label sticker. High contrast, maximum legibility, perfectly readable text physically on the packaging. "
        f"CULTURAL PATTERN: The outer background of the packaging is decorated with authentic {motif_name} traditional ethnic motif from {kabupaten}, {region}, South Sulawesi — "
        f"intricate geometric ornamental batik pattern framing the central white label cleanly. "
        f"MATERIAL: {material} material texture and finish. "
        f"{color_guide}"
        f"{badges}"
        f"STYLE: {market_style}, photorealistic 3D render, studio product photography, "
        f"isolated clean white studio background with soft drop shadow, glass reflections, sharp focus, 8K resolution, octane render quality. "
        f"CONTEXT: {description}"
    )
    return prompt.strip()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SulselPak AI Backend",
        "endpoints": [
            "GET /motifs",
            "GET /categories?type=makanan",
            "GET /materials",
            "POST /generate",
            "POST /api/generate-design"
        ]
    }


@app.post("/generate")
def generate_design(data: DesignRequest):
    warning_msg = None

    # Handle backward compatibility for `eco`
    if data.material is None and data.eco is not None:
        warning_msg = "Field 'eco' is deprecated. Please use 'material'."
        data.material = "Kertas Kraft" if data.eco else "Botol Plastik"
    elif data.material is None:
        data.material = "Kertas Kraft"

    # Validate material path if dataset exists
    try:
        resolve_material_path(data.material)
    except HTTPException:
        pass  # allow non-dataset material strings to pass through gracefully

    # Derive clean labels for prompt construction
    if data.category_subfolder:
        category_label = data.category_subfolder.replace("_", " ")
    elif data.category_filename:
        category_label = data.category_filename.rsplit('.', 1)[0].replace("_", " ")
    else:
        category_label = f"Produk {data.category_type.title()}"

    motif_raw = data.motif_filename.rsplit('.', 1)[0].replace("_", " ")
    if motif_raw.lower().startswith("motif "):
        motif_name = motif_raw[6:].strip()
    else:
        motif_name = motif_raw.strip()

    # Build full packaging prompt
    prompt = build_prompt(
        category_type=data.category_type,
        category_label=category_label,
        description=data.description,
        motif_name=motif_name,
        kabupaten=data.kabupaten,
        region=data.region,
        material=data.material,
    )

    # Resolve reference image based on priority rules (only motif/category refs, NOT sketch)
    # We skip user's sketch to avoid image-to-image mode that distorts the packaging output
    init_image_bytes, init_source = None, None
    if getattr(data, "use_motif_as_reference", False) or getattr(data, "use_category_as_reference", False):
        init_image_bytes, init_source = resolve_init_image(data)

    # Generate 4 alternative designs using Stability AI API
    generated_images = generate_stability_image(prompt, init_image_bytes=init_image_bytes, num_samples=4)

    # generated_images is a list[str]; provide backward-compat single field too
    first_image = generated_images[0] if generated_images else None

    response_payload = {
        "success": True,
        "images": generated_images,          # Array 4 base64 — untuk Flutter grid
        "image": first_image,                # Backward compat
        "image_base64": first_image,         # Backward compat
        "prompt": prompt,
        "prompt_used": prompt,
        "init_image_used": init_source,
    }
    if warning_msg:
        response_payload["warning"] = warning_msg

    return response_payload


@app.post("/api/generate-design")
async def generate_design_form(
    productName: str = Form(""),
    category: str = Form("Makanan"),
    product: str = Form(""),
    motif: str = Form(""),
    material: str = Form("Kertas Kraft"),
    targetMarket: str = Form("Lokal"),
    isHalal: str = Form("true"),
    halalCertNumber: str = Form(""),
    hasBPOM: str = Form("true"),
    bpomRegNumber: str = Form(""),
    nibNumber: str = Form(""),
    producerInfo: str = Form(""),
    storageInstructions: str = Form(""),
    netWeight: str = Form(""),
    expiryDate: str = Form(""),
    color: str = Form(""),
    enrichedPrompt: Optional[str] = Form(None),
    productImagePath: Optional[str] = Form(None),  # path gambar produk dari dataset
    sketch: Optional[UploadFile] = File(None),
):
    """
    Form-data endpoint compatible with ApiService.generateDesignComplete from Flutter.
    
    IMPORTANT: sketch is received but NOT used as init_image.
    Using sketch as init_image causes image-to-image mode which makes the AI
    reproduce the sketch structure instead of designing proper packaging.
    The prompt alone (text-to-image) produces much better packaging designs.
    """
    # Read sketch if provided (but we won't use it as init_image)
    _sketch_bytes = await sketch.read() if sketch else None

    # Resolve display name and motif
    prod_label = product if product else (productName if productName else category)
    motif_name = motif if motif else "Khas Sulsel Motif"
    halal_flag = isHalal.lower() in ("true", "1", "yes")

    if enrichedPrompt and enrichedPrompt.strip():
        # Use the enriched prompt from Flutter's PromptEngineeringService
        # But enhance it with explicit packaging instruction prefix
        base = enrichedPrompt.strip()
        prompt = (
            f"Professional commercial product PACKAGING DESIGN — "
            f"packaging box or container with label for '{prod_label}'. "
            f"DO NOT show the food product itself, show the PACKAGING design. "
            f"Label text: brand name '{productName or prod_label}' prominently displayed. "
            f"Cultural motif: {motif_name} traditional South Sulawesi pattern on label. "
            f"Material: {material}. "
            f"3D render, studio photography, white background, photorealistic, sharp focus, 8K. "
            f"Details: {base[:500]}"
        )
    else:
        description = f"South Sulawesi {category.lower()} product '{prod_label}', target market: {targetMarket}."
        if netWeight:
            description += f" Net weight: {netWeight}."
        if halal_flag:
            description += " Halal certified."

        prompt = build_prompt(
            category_type=category,
            category_label=prod_label,
            description=description,
            motif_name=motif_name,
            kabupaten="South Sulawesi",
            region="Sulawesi Selatan",
            material=material,
            product_name=productName,
            brand_name=productName,
            color_hint=color,
            is_halal=halal_flag,
            target_market=targetMarket,
        )

    print(f"[GENERATE] Product: {productName} | Motif: {motif_name} | Material: {material}")
    print(f"[GENERATE] Prompt preview: {prompt[:200]}...")

    # Resolve init_image from dataset product image if provided
    # This enables image-to-image mode for higher visual relevance to the actual product
    init_image_bytes: Optional[bytes] = None
    if productImagePath and productImagePath.strip():
        from app.services.dataset_repository import DATASET_DIR, validate_safe_path
        from fastapi import HTTPException as HEx
        try:
            decoded = productImagePath.strip().replace("%20", " ")
            target = DATASET_DIR / decoded
            safe_path = validate_safe_path(target)
            with open(safe_path, "rb") as f:
                init_image_bytes = f.read()
            print(f"[GENERATE] Using dataset product image as init_image: {decoded} ({len(init_image_bytes)} bytes)")
        except Exception as e:
            print(f"[GENERATE] Warning: Could not load product image '{productImagePath}': {e}")
            init_image_bytes = None

    # Generate 4 alternative designs
    # If init_image_bytes is set: image-to-image (higher product relevance)
    # If None: text-to-image (standard mode)
    generated_images = generate_stability_image(
        prompt,
        init_image_bytes=init_image_bytes,
        num_samples=4,
    )

    first_image = generated_images[0] if generated_images else None

    return {
        "success": True,
        "images": generated_images,          # Array 4 base64 — untuk Flutter grid
        "image_base64": first_image,         # Backward compat
        "image": first_image,                # Backward compat
        "prompt_used": prompt,
        "analysis": {
            "color_palette": ["#5D3A1A", "#D4AF37", "#FAF6F0"],
            "typography": f"Elegant serif font for brand name '{productName or prod_label}' dengan sentuhan kearifan lokal Sulsel",
            "layout": f"Kemasan {material} dengan motif {motif_name} sebagai elemen dekoratif label utama",
            "cultural_tips": f"Motif {motif_name} dari Sulawesi Selatan diintegrasikan sebagai pola border dan background label",
            "market_positioning": f"Desain premium untuk target pasar {targetMarket}",
            "umkm_compliance": "Pastikan label mencantumkan nama produk, berat bersih, tanggal kedaluwarsa, dan info produsen",
            "packaging_advantage": f"Material {material} memberikan tampilan premium dan ramah lingkungan",
        },
        "metadata": {
            "productName": productName,
            "category": category,
            "motif": motif,
            "material": material,
            "mode": "text_to_image",
            "sketch_received": _sketch_bytes is not None,
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    import os
    key = os.getenv("STABILITY_API_KEY", "")
    has_key = bool(key and key not in ("", "your_stability_api_key_here"))
    return {
        "status": "OK",
        "version": "3.0.0",
        "stability_api": "✅ Configured" if has_key else "❌ Missing STABILITY_API_KEY",
    }


@app.get("/debug/paths")
def debug_paths():
    """
    Debug endpoint: shows resolved DATASET_DIR, CWD, and lists top-level dataset folders.
    Useful for diagnosing image-not-found issues on HuggingFace Spaces.
    """
    import os
    from app.services.dataset_repository import DATASET_DIR
    from pathlib import Path

    cwd = str(Path.cwd())
    dataset_exists = DATASET_DIR.exists()
    dataset_is_dir = DATASET_DIR.is_dir() if dataset_exists else False

    subfolders = []
    if dataset_is_dir:
        try:
            subfolders = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
        except Exception as e:
            subfolders = [f"ERROR listing: {e}"]

    # Count images per subfolder
    counts = {}
    for folder in subfolders:
        try:
            fp = DATASET_DIR / folder
            counts[folder] = sum(1 for f in fp.rglob("*") if f.is_file())
        except Exception:
            counts[folder] = -1

    return {
        "cwd": cwd,
        "dataset_dir": str(DATASET_DIR),
        "dataset_exists": dataset_exists,
        "dataset_is_dir": dataset_is_dir,
        "subfolders": subfolders,
        "image_counts": counts,
        "env_DATASET_DIR": os.getenv("DATASET_DIR", "(not set)"),
    }
