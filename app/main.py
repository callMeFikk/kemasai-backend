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
    Returns (packaging_type_desc, packaging_shape) based on material (priority) then category.

    PRIORITY: Material keyword is checked FIRST. The `category_type` (e.g. "minuman")
    is only used as a fallback WITHIN the glass/bottle branch — and only when the
    material itself does not match any other known keyword. This prevents the old
    bug where every "minuman" category request became a glass jar regardless of
    the user-chosen material (e.g. Standup Pouch).

    KEYWORD SYNC: Keywords here are kept in sync with Flutter's
    PromptEngineeringService.generateEnrichedPrompt() — both sides must recognize
    the same set of terms so packaging shape is consistent whether enrichedPrompt
    is used or the backend falls back to build_prompt().

    UNKNOWN MATERIALS: If the material doesn't match any known keyword, we do NOT
    silently fall back to a kraft cardboard box. Instead we return a generic
    descriptor that still names the actual material, so the prompt never
    contradicts what the user requested.
    """
    cat = category_type.lower()
    mat = material.lower()

    def _matched(category_name: str, pack_type: str, pack_shape: str) -> tuple[str, str]:
        print(f"[MATERIAL] '{material}' → packaging_category='{category_name}'")
        return pack_type, pack_shape

    # ── 1. Stand-up pouch / plastik / sachet ─────────────────────────────────
    # Checked BEFORE glass/bottle so a Standup Pouch minuman stays a pouch.
    if ("pouch" in mat or "plastik" in mat or "zip" in mat
            or "standup" in mat or "standing" in mat or "sachet" in mat):
        return _matched(
            "standup_pouch",
            "hyperrealistic 3D commercial stand-up zip pouch food packaging bag",
            f"sealed matte finish zip-lock {material} pouch bag with transparent viewing window "
            f"and crisp front label printing"
        )

    # ── 2. Woven / natural fiber (lontar, bambu, rotan, anyaman, pelepah) ─────
    if ("lontar" in mat or "pisang" in mat or "pelepah" in mat
            or "anyaman" in mat or "bambu" in mat or "rotan" in mat
            or "woven" in mat):
        return _matched(
            "woven_natural_fiber",
            "hyperrealistic 3D artisan handwoven eco packaging container",
            f"artisan woven {material} container with woven lid and custom printed label tag sleeve"
        )

    # ── 3. Fabric / kain / tenun ──────────────────────────────────────────────
    if ("tenun" in mat or "kain" in mat or "fabric" in mat or "cloth" in mat):
        return _matched(
            "fabric_wrapped",
            "hyperrealistic 3D fabric-wrapped gift packaging box",
            f"elegant {material} wrapped gift box with printed ethnic label sleeve"
        )

    # ── 4. Ceramic / keramik / gerabah / porcelain ───────────────────────────
    if ("keramik" in mat or "ceramic" in mat
            or "porcelain" in mat or "gerabah" in mat):
        return _matched(
            "ceramic_jar",
            "hyperrealistic 3D ceramic jar container mockup with fitted lid",
            f"glazed ceramic {material} jar with fitted lid and printed front label sticker"
        )

    # ── 5. Metal / tin / aluminium / kaleng / foil ───────────────────────────
    if ("aluminium" in mat or "alumunium" in mat or "foil" in mat
            or "kaleng" in mat or "tin" in mat or "metal" in mat):
        return _matched(
            "metal_tin_can",
            "hyperrealistic 3D metal tin/aluminium can packaging mockup",
            f"metallic {material} tin can with printed wraparound label and crisp edges"
        )

    # ── 6. Paper / cardboard / kraft / karton ────────────────────────────────
    if ("kertas" in mat or "kraft" in mat or "karton" in mat
            or "kardus" in mat or "cardboard" in mat or "paper" in mat
            or "box" in mat or "kotak" in mat):
        return _matched(
            "cardboard_box",
            "hyperrealistic 3D kraft paper cardboard food packaging box",
            f"rectangular {material} food box with printed front label panel, crisp edges, studio lighting"
        )

    # ── 7. Glass / botol / kaca / jar ────────────────────────────────────────
    # This branch also activates as a FALLBACK when the category is "minuman"
    # and no other material keyword matched above. That way a user who picks
    # "Standup Pouch" for a minuman product stays a pouch (caught in branch 1),
    # but an unrecognized/generic minuman material still sensibly defaults to
    # a glass bottle rather than cardboard.
    if ("botol" in mat or "kaca" in mat or "glass" in mat or "jar" in mat
            or "minuman" in cat or "drink" in cat or "beverage" in cat):
        return _matched(
            "glass_jar_bottle",
            "hyperrealistic 3D commercial glass jar container mockup with shiny metallic gold screw lid",
            "transparent glass jar with realistic reflections, wrapped front label sticker"
        )

    # ── 8. Unknown material — preserve name, log warning ─────────────────────
    print(
        f"[WARN] Material '{material}' (category='{category_type}') did not match any known "
        f"packaging keyword. Using generic container description that preserves the material "
        f"name. If this material should map to a specific packaging shape, add its keyword "
        f"to _get_packaging_type() in main.py and the matching branch in "
        f"PromptEngineeringService.generateEnrichedPrompt() in Flutter."
    )
    return (
        f"hyperrealistic 3D commercial packaging container mockup made of {material}",
        f"custom packaging container made of {material}, with printed front label panel, "
        f"realistic texture and finish appropriate to {material}, crisp edges, studio lighting"
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
        f"MATERIAL: {material} material texture and finish. This is the ACTUAL packaging material — the container "
        f"shape and surface finish MUST visually match {material}, not any other material. "
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

    # Resolve reference image based on priority rules.
    #
    # FIX (updated): use_category_as_reference is still NOT honored — category/
    # product photos are pictures of food/drink, not packaging or motif texture,
    # and using them as an image-to-image reference was causing packaging-shape
    # mismatches.
    #
    # use_motif_as_reference IS honored again, but resolve_init_image() now
    # returns it with source "motif_image", which we use here to pick a very
    # LOW image_strength (MOTIF_IMAGE_STRENGTH) — enough to nudge color/texture
    # toward the real dataset motif artwork, without letting it override the
    # packaging shape/material the way the old 0.65 strength did.
    from app.services.stability_client import INIT_IMAGE_STRENGTH, MOTIF_IMAGE_STRENGTH

    init_image_bytes, init_source = None, None
    if getattr(data, "image", None):
        init_image_bytes, init_source = resolve_init_image(data)
    elif getattr(data, "use_motif_as_reference", False):
        init_image_bytes, init_source = resolve_init_image(data)
    elif getattr(data, "use_category_as_reference", False):
        print(
            "[INFO] use_category_as_reference requested but ignored: category/product "
            "photos are not valid packaging references and were causing packaging-shape "
            "mismatches. Falling back to text-to-image for this reference."
        )

    image_strength = MOTIF_IMAGE_STRENGTH if init_source == "motif_image" else INIT_IMAGE_STRENGTH
    print(
        f"[GENERATE /generate] material='{data.material}' init_source={init_source or 'none (text-to-image)'} "
        f"image_strength={image_strength if init_image_bytes else 'n/a'}"
    )

    # Generate 4 alternative designs using Stability AI API
    generated_images = generate_stability_image(
        prompt,
        init_image_bytes=init_image_bytes,
        num_samples=4,
        image_strength=image_strength,
    )

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

    FIX: productImagePath is likewise NOT used as init_image anymore. The dataset
    product image is a photo of the FOOD/DRINK product itself, not of packaging —
    using it as an image-to-image reference pulled the generated packaging shape
    and texture toward the food photo instead of the requested packaging material,
    causing material mismatches (e.g. asking for "Standup Pouch" but getting a shape
    that resembles the product photo). We now always use pure text-to-image here,
    and instead make sure the material is stated clearly and repeatedly in the prompt.
    """
    # Read sketch if provided (but we won't use it as init_image)
    _sketch_bytes = await sketch.read() if sketch else None

    # Resolve display name and motif
    prod_label = product if product else (productName if productName else category)
    motif_name = motif if motif else "Khas Sulsel Motif"
    halal_flag = isHalal.lower() in ("true", "1", "yes")

    # Resolve packaging shape for the prefix (material-first logic, same as _get_packaging_type)
    pack_type_prefix, _ = _get_packaging_type(category, material)

    if enrichedPrompt and enrichedPrompt.strip():
        # The enrichedPrompt from Flutter already states MATERIAL & SHAPE at its own
        # top line ("MATERIAL & SHAPE (MOST IMPORTANT — MUST MATCH EXACTLY): ...").
        # We prepend a SHORT backend prefix that also states material + shape so the
        # information appears TWICE at the very beginning of the final prompt —
        # critical because truncation from the END must never remove it.
        # The prefix is kept non-redundant with the negative_prompt to avoid
        # wasting the 2000-char budget on duplicated prohibitions.
        base = enrichedPrompt.strip()
        prompt = (
            f"PACKAGING MOCKUP. MATERIAL: {material}. "
            f"SHAPE: {pack_type_prefix}. "
            f"Show the container only — not the food or drink product itself.\n\n"
            f"{base}"
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

    prompt_len_before = len(prompt)
    print(f"[GENERATE] Product: {productName} | Category: {category} | Motif: {motif_name} | Material: {material}")
    print(f"[GENERATE] packaging_shape resolved: '{pack_type_prefix[:80]}...' (via _get_packaging_type)")
    print(f"[GENERATE] Prompt length before truncation: {prompt_len_before} chars")
    print(f"[GENERATE] Prompt preview (first 300 chars): {prompt[:300]}...")
    print("[DECISION] init_source=none (text-to-image only — productImagePath intentionally ignored; see docstring)")

    # NOTE: productImagePath is intentionally NOT loaded as init_image anymore.
    # The dataset product image is a photo of the FOOD/DRINK itself, not of
    # packaging — using it as image-to-image reference pulled the generated
    # packaging shape toward the food photo, causing material mismatches.
    if productImagePath and productImagePath.strip():
        print(
            f"[DECISION] productImagePath='{productImagePath}' received but IGNORED for "
            f"image-to-image to prevent packaging-material mismatch. "
            f"Using pure text-to-image. (Poin 1: foto produk bukan referensi kemasan)"
        )
    else:
        print("[DECISION] productImagePath not provided. Using pure text-to-image (no init_image).")
    init_image_bytes: Optional[bytes] = None

    # Generate 4 alternative designs (always text-to-image in this endpoint)
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
            "packaging_shape": pack_type_prefix,
            "mode": "text_to_image",
            "prompt_length": prompt_len_before,
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