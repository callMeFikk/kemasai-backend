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
        "clean modern minimal commercial retail packaging, high-contrast premium shelf appeal"
        if target_market.lower() in ("nasional", "national")
        else "authentic premium artisan local UMKM packaging, warm cultural heritage aesthetic, earthy tones"
    )

    prompt = (
        f"Award-winning 3D commercial product packaging mockup of {pack_type}. "
        f"PACKAGING SHAPE: {pack_shape}. "
        f"PHYSICAL FRONT LABEL: A crisp, solid white rectangular label sticker physically attached directly onto the front surface of the packaging. "
        f"PRODUCT TITLE ON PACKAGING: Large, ultra-sharp, bold dark typography titled '{display_name.upper()}' printed prominently in the center of the front packaging label sticker. High contrast, maximum legibility, perfectly readable text physically on the packaging. "
        f"CULTURAL PATTERN: The outer background of the packaging is decorated with authentic {motif_name} traditional ethnic motif from {kabupaten}, {region}, South Sulawesi — "
        f"seamlessly TILING, REPEATING geometric ornamental batik pattern at identical scale across ALL panels, framing the central white label cleanly. "
        f"MOTIF RULES: The pattern MUST tile perfectly at all fold edges — no shifting, no drift, no broken fragments. "
        f"MATERIAL: {material} material texture and finish. "
        f"{color_guide}"
        f"{badges}"
        f"STYLE: {market_style}, photorealistic 3D render, studio product photography, "
        f"isolated clean white studio background with soft drop shadow, sharp focus, 8K resolution, octane render quality. "
        f"STRICT: No food illustrations, no ingredient icons, no random floating elements. "
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


def build_detailed_analysis(
    productName: str,
    prod_label: str,
    category: str,
    motif_name: str,
    material: str,
    targetMarket: str,
    is_halal: bool,
    bpom_reg: str = "",
) -> dict:
    """
    Generates rich, detailed, structured cultural, aesthetic, and regulatory analysis.
    """
    m_lower = motif_name.lower()

    if any(k in m_lower for k in ["tedong", "kerbau"]):
        cultural_meaning = f"Motif '{motif_name}' melambangkan kemakmuran, derajat sosial tinggi, dan keteguhan prinsip dalam kebudayaan Toraja."
        colors = ["#8B0000", "#D4AF37", "#1A1A1A", "#FAF6F0"]
    elif any(k in m_lower for k in ["barre allo", "matahari"]):
        cultural_meaning = f"Motif '{motif_name}' melambangkan sumber kehidupan, kehangatan, keagungan, dan harapan tinggi bagi kemajuan usaha UMKM."
        colors = ["#D4AF37", "#E67E22", "#2C3E50", "#FFF8DC"]
    elif any(k in m_lower for k in ["sabbe", "lipa", "tenun"]):
        cultural_meaning = f"Motif '{motif_name}' terinspirasi dari keindahan tenun sutra Bugis yang melambangkan keanggunan, kehormatan, dan kehalusan budi."
        colors = ["#4A154B", "#D4AF37", "#8B1E3F", "#F5EBE0"]
    elif any(k in m_lower for k in ["sekong", "wajik", "bintik"]):
        cultural_meaning = f"Motif '{motif_name}' melambangkan ikatan kekeluargaan yang erat, persatuan, dan keteraturan ikhtiar."
        colors = ["#1B263B", "#C85A32", "#E0A96D", "#F4F1DE"]
    elif any(k in m_lower for k in ["balla", "somba", "gowa", "makassar"]):
        cultural_meaning = f"Motif '{motif_name}' merepresentasikan kejayaan dan kebanggaan kebudayaan maritim Makassar & Gowa."
        colors = ["#A31D1D", "#D4AF37", "#0F4C81", "#FAF7F2"]
    else:
        cultural_meaning = f"Motif '{motif_name}' menyimbolkan nilai estetika kearifan lokal Sulawesi Selatan yang memberikan identitas etnik kuat pada produk."
        colors = ["#5D3A1A", "#D4AF37", "#2C3E50", "#FAF6F0"]

    t_lower = targetMarket.lower()
    if "muda" in t_lower or "gen-z" in t_lower or "millennial" in t_lower:
        strategy = f"Dikemas modern-minimalis untuk memikat konsumen {targetMarket}. Warna kontras dan motif {motif_name} tampil estetik (instagramable) tanpa mengesampingkan identitas etnik."
    elif "oleh" in t_lower or "wisata" in t_lower or "turis" in t_lower:
        strategy = f"Fokus utama sebagai produk oleh-oleh khas Sulsel berkesan premium. Motif {motif_name} ditonjolkan sebagai daya tarik budaya autentik yang berkesan tinggi."
    elif "ekspor" in t_lower or "global" in t_lower:
        strategy = f"Standar visual internasional untuk pasar ekspor. Kemasan {material} dengan aksen motif {motif_name} memberikan keunggulan produk etnik nusantara berkualitas tinggi."
    else:
        strategy = f"Posisi produk sebagai pilihan utama bagi {targetMarket} melalui perpaduan aksen etnik khas Sulawesi Selatan dan tampilan kemasan {material} yang rapi & profesional."

    compliance_items = []
    if is_halal:
        compliance_items.append("Logo Halal Indonesia tercantum jelas di bagian depan kemasan.")
    if bpom_reg:
        compliance_items.append(f"Nomor Izin BPOM ({bpom_reg}) wajib dicantumkan pada label informasi.")
    else:
        compliance_items.append("Cantumkan izin P-IRT / BPOM pada panel legalitas kemasan.")
    compliance_items.append("Wajib mencantumkan Nama Produk, Berat Bersih (Netto), Tanggal Kadaluarsa, dan Info Produsen.")

    return {
        "color_palette": colors,
        "typography": f"Font judul brand '{productName or prod_label}' menggunakan serif/sans-serif elegan beraksen emas/kontras, sedangkan teks legal menggunakan font bersih sans-serif yang mudah dibaca.",
        "layout": f"Tata letak simetris fokus tengah: Nama Brand '{productName or prod_label}' sebagai pusat perhatian utama, dipadukan bingkai dekoratif motif {motif_name} di area background label kemasan {material}.",
        "cultural_tips": cultural_meaning,
        "market_positioning": strategy,
        "umkm_compliance": " | ".join(compliance_items),
        "packaging_advantage": f"Material {material} memberikan perlindungan optimal terhadap kelembapan dan kebersihan, menjaga kualitas khas {prod_label}, serta meningkatkan daya tarik visual produk.",
    }


@app.post("/api/generate-design")
async def generate_design_complete(
    productName: str = Form(""),
    category: str = Form("makanan"),
    product: str = Form(""),
    motif: str = Form(""),
    material: str = Form("Standup Pouch"),
    targetMarket: str = Form("Semua Usia"),
    isHalal: str = Form("false"),
    halalCertNumber: str = Form(""),
    hasBPOM: str = Form("false"),
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
        # FIX E: Kirim FULL enrichedPrompt dari Flutter tanpa truncation.
        # Backend HANYA menambahkan prefix teknis ringan dan suffix — tidak memotong
        # atau menimpa konten kaya (motif, warna, identitas dataset) dari PromptEngineeringService.
        base = enrichedPrompt.strip()
        prompt = (
            f"Commercial PACKAGING DESIGN MOCKUP — professional print-ready product packaging "
            f"for '{productName or prod_label}' ({category}). "
            f"Material: {material}. "
            f"Show ONLY the packaging container/box, NOT the food product itself. "
            f"3D render, studio photography, clean white background, photorealistic, sharp focus, 8K. "
            f"\n\n{base}"
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
        "analysis": build_detailed_analysis(
            productName=productName,
            prod_label=prod_label,
            category=category,
            motif_name=motif_name,
            material=material,
            targetMarket=targetMarket,
            is_halal=halal_flag,
            bpom_reg=bpomRegNumber,
        ),
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
