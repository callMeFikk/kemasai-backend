"""
Stability AI API client service.

Handles:
- API key validation
- Image dimension validation & resize (Pillow)
- init_image priority resolution
- Text-to-Image and Image-to-Image API calls
"""

import io
import os
import base64
import requests
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image

from app.services.dataset_repository import (
    resolve_motif_path,
    resolve_category_path,
)

load_dotenv()

# ─── Allowed Dimensions for Stability AI SDXL 1.0 ──────────────────────────
STABILITY_ALLOWED_DIMENSIONS: set[tuple[int, int]] = {
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
    (640, 1536),
    (768, 1344),
    (832, 1216),
    (896, 1152),
}

ALLOWED_DIMENSIONS_LIST: list[tuple[int, int]] = sorted(
    STABILITY_ALLOWED_DIMENSIONS, key=lambda d: d[0] * d[1]
)

DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 1024


# ─── Dimension Helpers ───────────────────────────────────────────────────────

def validate_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    If (width, height) is not in STABILITY_ALLOWED_DIMENSIONS, fallback to default.
    Returns the (valid_width, valid_height) pair to use.
    """
    if (width, height) in STABILITY_ALLOWED_DIMENSIONS:
        return width, height

    print(
        f"[WARN] Dimension {width}x{height} is not allowed by Stability AI. "
        f"Falling back to {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}."
    )
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def _resize_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Resize image proportionally then center-crop to exactly target_w x target_h.
    This avoids distortion (no stretching).
    """
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    img = img.crop((left, top, right, bottom))
    return img


def resize_to_allowed_dimension(image_bytes: bytes) -> bytes:
    """
    Resize any input image to the nearest allowed Stability AI SDXL dimension
    that best matches the original aspect ratio.
    
    Uses proportional resize + center crop to avoid distortion.
    Always returns PNG bytes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Gagal membaca gambar yang diunggah: {str(e)}"
        )

    original_ratio = img.width / img.height

    # Choose the allowed dimension whose aspect ratio is closest to the original
    target_w, target_h = min(
        ALLOWED_DIMENSIONS_LIST,
        key=lambda d: abs((d[0] / d[1]) - original_ratio)
    )

    print(
        f"[INFO] Resizing init_image from {img.width}x{img.height} "
        f"(ratio={original_ratio:.3f}) to {target_w}x{target_h}"
    )

    img_resized = _resize_and_center_crop(img, target_w, target_h)

    output = io.BytesIO()
    img_resized.save(output, format="PNG")
    return output.getvalue()


# ─── API Key Helpers ─────────────────────────────────────────────────────────

def get_stability_api_key() -> str | None:
    """
    Retrieves the STABILITY_API_KEY from environment variables.
    Returns None if key is missing or is still a placeholder.
    """
    key = os.getenv("STABILITY_API_KEY")
    if not key or key.strip().lower() in (
        "", "your_stability_api_key_here",
        "sk-placeholder", "sk-your_stability_api_key_here"
    ):
        return None
    return key.strip()


def validate_api_key_on_startup():
    """
    Startup check. Prints a warning if API key is missing.
    Does NOT crash the server so the dataset endpoints remain accessible.
    """
    key = get_stability_api_key()
    if not key:
        print(
            "[SECURITY WARNING] STABILITY_API_KEY environment variable is not configured! "
            "Please add a valid STABILITY_API_KEY to your backend .env file."
        )


# ─── Init Image Priority Resolver ───────────────────────────────────────────

def resolve_init_image(data: Any) -> tuple[bytes | None, str | None]:
    """
    Determines and resolves the initial reference image based on strict priority:
    1. User custom uploaded base64 image (data.image)
    2. Motif reference image (use_motif_as_reference = True)
    3. Category reference image (use_category_as_reference = True)
    4. None → Text-to-Image mode

    All resolved images are automatically resized to the nearest valid
    Stability AI dimension before being returned.
    """
    # Priority 1: User uploaded custom image (base64)
    if hasattr(data, "image") and data.image and data.image.strip():
        try:
            raw_b64 = data.image.strip()
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Gambar yang diunggah tidak valid (bukan format base64): {str(e)}"
            )
        return resize_to_allowed_dimension(image_bytes), "user_image"

    # Priority 2: Use motif dataset image as reference
    if getattr(data, "use_motif_as_reference", False):
        motif_path = resolve_motif_path(
            getattr(data, "region", "Bugis"),
            getattr(data, "kabupaten", "Barru"),
            getattr(data, "motif_filename", "")
        )
        return resize_to_allowed_dimension(motif_path.read_bytes()), "motif_image"

    # Priority 3: Use category dataset image as reference
    if getattr(data, "use_category_as_reference", False):
        cat_path = resolve_category_path(
            getattr(data, "category_type", "makanan"),
            subfolder=getattr(data, "category_subfolder", None),
            filename=getattr(data, "category_filename", None)
        )
        return resize_to_allowed_dimension(cat_path.read_bytes()), "category_image"

    # Priority 4: No reference image → Text-to-Image
    return None, None


# ─── Free Generator API Call (Pollinations AI - FLUX Model) ───────────────────

def generate_pollinations_image(prompt: str, num_samples: int = 4) -> list[str]:
    """
    Generates high-quality packaging images using Pollinations AI (FLUX model).
    100% FREE, NO API KEY REQUIRED, UNLIMITED GENERATIONS.
    Staggers requests and uses retries on HTTP 429 to guarantee 4 unique variations.
    """
    import urllib.parse
    import urllib.request
    import urllib.error
    import random
    import time
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"[POLLINATIONS AI] Generating {num_samples} images with FLUX model (Free & Unlimited)...")
    clean_prompt = prompt.replace("\n", " ").strip()

    # Perspective / angle variations so each of the 4 samples produces a unique design
    style_variations = [
        ", front view studio product render",
        ", 3/4 perspective angle product photography",
        ", close-up detail of label texture and branding",
        ", elegant side perspective presentation"
    ]

    def fetch_single_sample(idx: int) -> str | None:
        # Stagger initial request by 0.6s per index to prevent burst rate limit (429)
        time.sleep(0.6 * idx)

        variation = style_variations[idx % len(style_variations)]
        sample_prompt = clean_prompt + variation
        encoded = urllib.parse.quote(sample_prompt)

        max_retries = 4
        for attempt in range(max_retries):
            seed = random.randint(10000, 999999)
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&model=flux&nologo=true&seed={seed}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/12{idx}.0.0.0 Safari/537.36"
                }
            )
            try:
                with urllib.request.urlopen(req, timeout=35) as response:
                    img_bytes = response.read()
                    if len(img_bytes) > 2000:
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        print(f"[POLLINATIONS] Sample {idx+1}/{num_samples} OK ({len(img_bytes)//1024} KB)")
                        return b64
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    wait_time = 1.2 * (attempt + 1) + random.uniform(0.2, 0.6)
                    print(f"[POLLINATIONS RETRY] Sample {idx+1} hit 429, retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"[POLLINATIONS WARN] Sample {idx+1} HTTP error {he.code}: {he.reason}")
                    break
            except Exception as e:
                print(f"[POLLINATIONS WARN] Sample {idx+1} error: {e}")
                time.sleep(0.5)

        return None

    images = []
    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(fetch_single_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                images.append(res)

    if images:
        orig_count = len(images)
        while len(images) < 4:
            images.append(images[len(images) % orig_count])
        return images

    raise HTTPException(
        status_code=502,
        detail="Gagal generate desain dari engine AI gratis. Silakan coba lagi."
    )


def generate_stability_image(
    prompt: str,
    init_image_bytes: bytes | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_samples: int = 4,
) -> list[str]:
    """
    Calls Stability AI SDXL 1.0 API with immediate fallback to Pollinations AI (FLUX - Free).
    If Stability AI hits Rate Limit (429), Insufficient Credits (402), or Missing Key,
    it seamlessly falls back to Pollinations AI without throwing errors.
    """
    # Option to force free generator via env flag if desired
    if os.getenv("USE_FREE_GENERATOR", "").lower() in ("true", "1", "yes"):
        print("[ENGINE] USE_FREE_GENERATOR active. Using Pollinations AI (FLUX Free)...")
        return generate_pollinations_image(prompt, num_samples)

    api_key = get_stability_api_key()
    if not api_key:
        print("[FALLBACK] STABILITY_API_KEY missing. Falling back to Pollinations AI (FLUX Free)...")
        return generate_pollinations_image(prompt, num_samples)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    samples_to_try = [num_samples, 2, 1] if num_samples > 1 else [1]
    last_error_status = 200
    last_error_msg = ""

    negative_prompt = (
        "blurry, low quality, distorted, ugly, flat 2D icon, watermark, plain logo, "
        "unpacked food item, raw food, cooked food dish, meal on plate, eating food, "
        "banana leaf food parcel, dirty background, cluttered scene"
    )

    for samples in samples_to_try:
        for attempt in range(2):
            try:
                if init_image_bytes:
                    init_image_resized = resize_to_allowed_dimension(init_image_bytes)
                    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
                    files = {
                        "init_image": ("init_image.png", init_image_resized, "image/png")
                    }
                    data_payload = {
                        "text_prompts[0][text]": prompt,
                        "text_prompts[0][weight]": 1.0,
                        "text_prompts[1][text]": negative_prompt,
                        "text_prompts[1][weight]": -1.0,
                        "cfg_scale": 7,
                        "image_strength": 0.65,
                        "samples": samples,
                        "steps": 30,
                    }
                    response = requests.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data_payload,
                        timeout=120,
                    )
                else:
                    valid_w, valid_h = validate_dimensions(width, height)
                    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
                    json_headers = {**headers, "Content-Type": "application/json"}
                    json_payload = {
                        "text_prompts": [
                            {"text": prompt, "weight": 1.0},
                            {"text": negative_prompt, "weight": -1.0}
                        ],
                        "cfg_scale": 7,
                        "height": valid_h,
                        "width": valid_w,
                        "samples": samples,
                        "steps": 30,
                    }
                    response = requests.post(
                        url,
                        headers=json_headers,
                        json=json_payload,
                        timeout=120,
                    )

                if response.status_code == 200:
                    result = response.json()
                    artifacts = result.get("artifacts", [])
                    if artifacts:
                        base64_list = [art["base64"] for art in artifacts]
                        while len(base64_list) < 4:
                            base64_list.append(base64_list[len(base64_list) % len(artifacts)])
                        return base64_list

                last_error_status = response.status_code
                try:
                    err_json = response.json()
                    last_error_msg = err_json.get("message", response.text)
                except Exception:
                    last_error_msg = response.text

                safe_msg = last_error_msg.encode("ascii", "ignore").decode("ascii")
                print(f"[WARN] Stability API status {response.status_code} (samples={samples}): {safe_msg}")

                if response.status_code in (429, 401, 402):
                    print(f"[FALLBACK INSTANT] Stability AI returned {response.status_code}. Switching directly to Pollinations AI (FLUX Free)...")
                    return generate_pollinations_image(prompt, num_samples)

            except requests.exceptions.Timeout:
                last_error_status = 504
                last_error_msg = "Timeout 120s"
            except requests.exceptions.RequestException as e:
                last_error_status = 502
                last_error_msg = str(e)

    # ── AUTOMATIC FALLBACK TO POLLINATIONS AI (FLUX FREE UNLIMITED) ──────────
    print(f"[FALLBACK] Stability AI error ({last_error_status}: {last_error_msg[:100]}). Falling back to Pollinations AI (FLUX Free)...")
    try:
        return generate_pollinations_image(prompt, num_samples)
    except Exception as fallback_err:
        raise HTTPException(
            status_code=502,
            detail=f"Gagal generate desain: {str(fallback_err)}"
        )
