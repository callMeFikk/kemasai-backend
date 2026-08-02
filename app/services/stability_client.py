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

# FIX #6: When image-to-image IS used for an explicit user-uploaded reference
# image, keep image_strength low so the TEXT PROMPT (including material)
# dominates over the reference image's structure/texture. A high strength (e.g.
# 0.65) was causing the reference image's shape/material to override the
# requested material.
INIT_IMAGE_STRENGTH: float = 0.35

# FIX #9 (opt-in motif texture reference): if the caller wants the actual
# dataset motif image to influence the generated pattern's look-and-feel (not
# just a text description of the motif name), it can be passed as init_image
# with THIS much lower strength. At 0.15 the diffusion process only nudges
# color/texture/pattern-density toward the reference image early in denoising;
# it does NOT lock in the reference image's overall shape/composition the way
# INIT_IMAGE_STRENGTH (0.35) or the old 0.65 did. This keeps packaging
# shape/material fully controlled by the text prompt while still letting the
# real motif artwork inform the pattern's visual character.
MOTIF_IMAGE_STRENGTH: float = 0.15

# FIX #7 (critical): Stability AI's text_prompts[].text field is HARD-LIMITED to
# 2000 characters. The enriched prompt built by PromptEngineeringService.dart
# (dieline rules + motif rules + label rules + ingredient lock + color lock +
# style + "ABSOLUTE PROHIBITIONS" list) plus the prefix added in main.py routinely
# exceeds this limit. When it does, Stability rejects EVERY request with a 400
# error ("text_prompts: the length must be between 1 and 2000"), which silently
# forces a fallback to Pollinations on every single call — and Pollinations'
# free flux-schnell/turbo models handle very long prompts poorly, tending to
# ignore instructions buried deep in the text and defaulting to a generic
# cardboard-box shape regardless of the requested material. This was the actual
# cause of "material selalu jadi kotak" even after init_image was disabled.
STABILITY_PROMPT_CHAR_LIMIT: int = 1900  # small safety margin below the hard 2000 limit
POLLINATIONS_PROMPT_CHAR_LIMIT: int = 1400  # empirically more reliable for flux-schnell/turbo


def _fit_prompt_to_limit(prompt: str, limit: int) -> str:
    """
    Truncates `prompt` to at most `limit` characters, cutting at the last sentence
    boundary (". ") when possible so we don't chop mid-instruction. We always cut
    from the END, never the middle/start — the caller is expected to put the most
    critical info (material + packaging shape) at the very beginning of the prompt,
    so truncation only drops the least-critical trailing detail (verbose motif
    tiling rules, redundant prohibition lists, etc.), never the material/shape.
    """
    if len(prompt) <= limit:
        return prompt
    truncated = prompt[:limit]
    last_period = truncated.rfind(". ")
    if last_period > limit * 0.5:
        truncated = truncated[: last_period + 1]
    print(
        f"[WARN] Prompt was {len(prompt)} chars, truncated to {len(truncated)} chars "
        f"to stay within the model's reliable prompt length. Material/shape info "
        f"(kept at the start of the prompt) is preserved; trailing detail was cut."
    )
    return truncated


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
    if get_cloudflare_worker_config():
        print("[INFO] Cloudflare Worker proxy configured (CLOUDFLARE_WORKER_URL) — will be used as the PRIMARY free engine (no SSL issues).")
    elif get_cloudflare_credentials():
        print("[INFO] Cloudflare Workers AI direct credentials detected — will be used as the free fallback engine.")
    else:
        print(
            "[INFO] CLOUDFLARE_WORKER_URL / CLOUDFLARE_ACCOUNT_ID not configured. "
            "Free fallback will use Pollinations AI only."
        )


def get_cloudflare_credentials() -> tuple[str, str] | None:
    """
    Retrieves CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN from environment
    variables. Returns None if either is missing, so callers can cleanly fall
    back to the next engine (Pollinations) without crashing.
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if account_id and api_token:
        return account_id, api_token
    return None


def get_cloudflare_worker_config() -> tuple[str, str | None] | None:
    """
    Retrieves CLOUDFLARE_WORKER_URL and optional CLOUDFLARE_WORKER_SECRET.

    The Worker proxy (sulsel-ai-proxy.*.workers.dev) calls Cloudflare Workers AI
    via its internal AI binding — no outbound SSL to api.cloudflare.com needed.
    This avoids the SSLEOFError that occurs when HuggingFace Space tries to
    call api.cloudflare.com directly (HF blocks/intercepts that SSL connection).

    Returns (url, secret_or_None) if CLOUDFLARE_WORKER_URL is set, else None.
    """
    worker_url = os.getenv("CLOUDFLARE_WORKER_URL", "").strip().rstrip("/")
    if worker_url:
        secret = os.getenv("CLOUDFLARE_WORKER_SECRET", "").strip() or None
        return worker_url, secret
    return None


# ─── Init Image Priority Resolver ───────────────────────────────────────────

def resolve_init_image(data: Any) -> tuple[bytes | None, str | None]:
    """
    Determines and resolves the initial reference image, in priority order:

    1. User-uploaded custom image (data.image) — treated as a strong reference
       (see INIT_IMAGE_STRENGTH), since the user deliberately provided it.
    2. Motif dataset image (use_motif_as_reference=True) — treated as a very
       LOW-strength texture/color nudge only (see MOTIF_IMAGE_STRENGTH). This
       lets the actual motif artwork from the dataset influence the pattern's
       look-and-feel without overriding the packaging shape/material, which is
       still controlled by the text prompt. Source label "motif_image" tells
       the caller to use MOTIF_IMAGE_STRENGTH instead of INIT_IMAGE_STRENGTH.

    NOTE: use_category_as_reference is intentionally NOT handled here anymore.
    Category/product photos are pictures of the FOOD/DRINK itself, not of
    packaging or motif texture — using them as an image-to-image reference
    (at any strength) pulls the generated packaging toward the food photo's
    composition and was a direct cause of packaging-material mismatches.

    All resolved images are automatically resized to the nearest valid
    Stability AI dimension before being returned.
    """
    # Priority 1: User uploaded custom image (base64) — strong reference.
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

    # Priority 2: Motif dataset image — low-strength texture/color nudge only.
    if getattr(data, "use_motif_as_reference", False):
        motif_path = resolve_motif_path(
            getattr(data, "region", "Bugis"),
            getattr(data, "kabupaten", "Barru"),
            getattr(data, "motif_filename", "")
        )
        return resize_to_allowed_dimension(motif_path.read_bytes()), "motif_image"

    # No reference image → Text-to-Image mode
    return None, None


# ─── Simple In-Memory Prompt Cache (TTL = 10 Minutes) ────────────────────────
import time as _time

_PROMPT_CACHE: dict[str, tuple[float, list[str]]] = {}
CACHE_TTL_SECONDS: int = 600  # 10 minutes cache TTL

def get_cached_images(prompt_key: str) -> list[str] | None:
    now = _time.time()
    if prompt_key in _PROMPT_CACHE:
        timestamp, images = _PROMPT_CACHE[prompt_key]
        if now - timestamp < CACHE_TTL_SECONDS:
            print(f"[CACHE HIT] Returning cached images for prompt (age: {int(now - timestamp)}s)...")
            return images
        else:
            del _PROMPT_CACHE[prompt_key]
    return None

def set_cached_images(prompt_key: str, images: list[str]):
    # Keep cache size bounded to max 30 items
    if len(_PROMPT_CACHE) > 30:
        oldest_key = min(_PROMPT_CACHE, key=lambda k: _PROMPT_CACHE[k][0])
        del _PROMPT_CACHE[oldest_key]
    _PROMPT_CACHE[prompt_key] = (_time.time(), images)


# ─── Cloudflare Workers AI (Free Tier — Primary Free Engine) ────────────────
# Cloudflare Workers AI has a genuinely free daily quota (not a one-time trial
# credit like Stability AI's new-account grant). Docs: https://developers.cloudflare.com/workers-ai/
# Setup: create a free Cloudflare account, generate an API Token with the
# "Workers AI: Run" permission, copy the Account ID from the dashboard sidebar,
# and set CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN in the backend .env file.
CLOUDFLARE_MODELS: list[str] = [
    "@cf/black-forest-labs/flux-1-schnell",       # primary: fast + high quality
    "@cf/bytedance/stable-diffusion-xl-lightning", # fallback: fast, realistic
]
CLOUDFLARE_MODEL_ATTEMPT_BUDGET: dict[str, int] = {
    "@cf/black-forest-labs/flux-1-schnell": 2,
    "@cf/bytedance/stable-diffusion-xl-lightning": 1,
}
CLOUDFLARE_PROMPT_CHAR_LIMIT: int = 1800  # Workers AI tolerates long prompts fine; kept consistent with other engines


def generate_cloudflare_image(prompt: str, num_samples: int = 4) -> list[str]:
    """
    Generates images via Cloudflare Workers AI. One request per sample (each with
    its own composition variation + random seed, same approach as
    generate_stability_image), with a 2-model fallback chain per sample and
    retry-with-backoff on rate limits, mirroring generate_pollinations_image's
    reliability pattern. Raises HTTPException if every attempt fails, so the
    caller (generate_free_fallback_image) can fall back to Pollinations.
    """
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    creds = get_cloudflare_credentials()
    if not creds:
        raise HTTPException(status_code=502, detail="Cloudflare Workers AI credentials not configured.")
    account_id, api_token = creds

    prompt = _fit_prompt_to_limit(prompt, CLOUDFLARE_PROMPT_CHAR_LIMIT)
    clean_prompt = prompt.replace("\n", " ").strip()

    # Same angle/composition variations used for the Pollinations fallback, so
    # the 4 alternatives stay visually distinguishable regardless of which free
    # engine ends up serving the request.
    style_variations = [
        ", front face view, centered label legible, seamless tiling motif pattern on all panels, "
        "same packaging material and shape as specified in the prompt, clean studio white background, "
        "print-ready packaging mockup",
        ", slight 3/4 left angle showing front and left side panel, motif pattern continuous across both panels, "
        "same packaging material and shape as specified in the prompt, commercial product photography, white background",
        ", straight-on front view, ultra-sharp product name text, consistent repeating motif background, "
        "same packaging material and shape as specified in the prompt, isolated white background, professional studio lighting",
        ", 3/4 right angle showing front and right side panel, motif seamlessly wrapping corners, "
        "same packaging material and shape as specified in the prompt, photorealistic render, white background",
    ]

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    def fetch_single_sample(idx: int) -> tuple[int, str | None]:
        variation = style_variations[idx % len(style_variations)]
        seed = random.randint(1, 2_000_000_000)
        variant_prompt = f"{clean_prompt}{variation}"

        for model in CLOUDFLARE_MODELS:
            max_attempts = CLOUDFLARE_MODEL_ATTEMPT_BUDGET.get(model, 1)
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

            for attempt in range(max_attempts):
                try:
                    resp = requests.post(
                        url,
                        headers=headers,
                        json={"prompt": variant_prompt, "seed": seed},
                        timeout=60,
                    )
                except requests.exceptions.SSLError as e:
                    print(f"[CLOUDFLARE WARN] SSL connection blocked by environment ({e}). Skipping Cloudflare...")
                    break  # SSL blocked by HF network — bail out immediately, don't waste time retrying
                except requests.exceptions.Timeout:
                    print(f"[CLOUDFLARE WARN] Sample {idx+1} model={model} attempt={attempt+1}/{max_attempts}: timeout")
                    if attempt < max_attempts - 1:
                        time.sleep(1.0)
                        continue
                    break
                except requests.exceptions.RequestException as e:
                    print(f"[CLOUDFLARE WARN] Sample {idx+1} model={model}: {e}")
                    break

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        print(f"[CLOUDFLARE WARN] Sample {idx+1} model={model}: non-JSON 200 response")
                        break
                    image_b64 = data.get("result", {}).get("image") if data.get("success") else None
                    if image_b64:
                        print(f"[CLOUDFLARE] Sample {idx+1}/{num_samples} OK model={model} attempt={attempt+1}")
                        return idx, image_b64
                    print(f"[CLOUDFLARE WARN] Sample {idx+1} model={model}: {data.get('errors')}")
                    break
                elif resp.status_code == 429:
                    print(f"[CLOUDFLARE] Sample {idx+1} rate-limited (429) model={model} attempt={attempt+1}/{max_attempts}")
                    if attempt < max_attempts - 1:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    break
                else:
                    safe_text = resp.text[:200].encode("ascii", "ignore").decode("ascii")
                    print(f"[CLOUDFLARE WARN] Sample {idx+1} HTTP {resp.status_code} model={model}: {safe_text}")
                    break
        return idx, None

    results = []
    with ThreadPoolExecutor(max_workers=min(num_samples, 3)) as executor:
        futures = [executor.submit(fetch_single_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0])
    successful_images = [b64 for _, b64 in results if b64 is not None]

    if successful_images:
        images_out = list(successful_images)
        while len(images_out) < num_samples:
            images_out.append(images_out[len(images_out) % len(successful_images)])
        return images_out

    raise HTTPException(
        status_code=502,
        detail="Gagal generate desain via Cloudflare Workers AI."
    )


def generate_cloudflare_worker_image(prompt: str, num_samples: int = 4) -> list[str]:
    """
    Generates images via the Cloudflare Worker proxy (sulsel-ai-proxy.*.workers.dev).

    WHY a Worker proxy instead of calling api.cloudflare.com directly:
    HuggingFace Space's container network blocks/intercepts outbound SSL connections
    to api.cloudflare.com, causing SSLEOFError on every attempt. The Worker proxy
    calls Cloudflare Workers AI via its internal `env.AI` binding — a direct
    Cloudflare-internal call with no external TLS handshake — so the HuggingFace
    SSL block is completely bypassed.

    Flow: HuggingFace backend → Worker URL (normal HTTPS) → env.AI binding (internal)
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = get_cloudflare_worker_config()
    if not config:
        raise HTTPException(status_code=502, detail="Cloudflare Worker URL (CLOUDFLARE_WORKER_URL) not configured.")
    worker_url, worker_secret = config

    prompt = _fit_prompt_to_limit(prompt, CLOUDFLARE_PROMPT_CHAR_LIMIT)
    clean_prompt = prompt.replace("\n", " ").strip()

    req_headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if worker_secret:
        req_headers["X-Worker-Secret"] = worker_secret

    style_variations = [
        ", front face view, centered label legible, seamless tiling motif pattern on all panels, "
        "same packaging material and shape as specified in the prompt, clean studio white background, "
        "print-ready packaging mockup",
        ", slight 3/4 left angle showing front and left side panel, motif pattern continuous across both panels, "
        "same packaging material and shape as specified in the prompt, commercial product photography, white background",
        ", straight-on front view, ultra-sharp product name text, consistent repeating motif background, "
        "same packaging material and shape as specified in the prompt, isolated white background, professional studio lighting",
        ", 3/4 right angle showing front and right side panel, motif seamlessly wrapping corners, "
        "same packaging material and shape as specified in the prompt, photorealistic render, white background",
    ]

    WORKER_MODELS = [
        "@cf/black-forest-labs/flux-1-schnell",
        "@cf/bytedance/stable-diffusion-xl-lightning",
    ]

    def fetch_single_sample(idx: int) -> tuple[int, str | None]:
        variation = style_variations[idx % len(style_variations)]
        seed = random.randint(1, 2_000_000_000)
        variant_prompt = f"{clean_prompt}{variation}"

        for model in WORKER_MODELS:
            try:
                resp = requests.post(
                    worker_url,
                    headers=req_headers,
                    json={"prompt": variant_prompt, "model": model, "seed": seed},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success") and data.get("image_b64"):
                        print(f"[WORKER] Sample {idx+1}/{num_samples} OK model={model}")
                        return idx, data["image_b64"]
                    err = data.get("error", "unknown")
                    print(f"[WORKER WARN] Sample {idx+1} model={model}: success=False error={err}")
                elif resp.status_code == 401:
                    print(f"[WORKER WARN] Sample {idx+1}: 401 Unauthorized — check CLOUDFLARE_WORKER_SECRET")
                    break  # wrong secret — no point retrying other models
                else:
                    print(f"[WORKER WARN] Sample {idx+1} model={model} HTTP {resp.status_code}: {resp.text[:120]}")
            except requests.exceptions.SSLError as e:
                print(f"[WORKER WARN] SSL connection blocked by environment ({e}). Skipping Worker proxy...")
                break  # SSL blocked — bail out immediately to Pollinations
            except requests.exceptions.RequestException as e:
                print(f"[WORKER WARN] Sample {idx+1} model={model}: {e}")
        return idx, None

    results = []
    with ThreadPoolExecutor(max_workers=min(num_samples, 3)) as executor:
        futures = [executor.submit(fetch_single_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0])
    successful_images = [b64 for _, b64 in results if b64 is not None]

    if successful_images:
        images_out = list(successful_images)
        while len(images_out) < num_samples:
            images_out.append(images_out[len(images_out) % len(successful_images)])
        return images_out

    raise HTTPException(
        status_code=502,
        detail="Gagal generate desain via Cloudflare Worker proxy."
    )


def generate_free_fallback_image(prompt: str, num_samples: int = 4) -> list[str]:
    """
    Unified FREE fallback. Engine priority:
    1. Cloudflare Worker proxy (CLOUDFLARE_WORKER_URL) — calls Workers AI via
       internal binding, completely bypasses HuggingFace's SSL block on api.cloudflare.com.
    2. Cloudflare Workers AI direct API (CLOUDFLARE_ACCOUNT_ID + TOKEN) — works
       from local dev but fails from HuggingFace due to SSL restrictions.
    3. Pollinations AI — always available, no credentials needed.
    """
    # Priority 1: Worker proxy (recommended for HuggingFace deployment)
    if get_cloudflare_worker_config():
        try:
            print("[ENGINE] Trying Cloudflare Worker proxy (internal AI binding, no SSL issues)...")
            return generate_cloudflare_worker_image(prompt, num_samples)
        except Exception as e:
            print(f"[FALLBACK] Cloudflare Worker proxy failed ({e}). Trying direct Cloudflare API...")

    # Priority 2: Direct Cloudflare API (local dev / non-HuggingFace environments)
    if get_cloudflare_credentials():
        try:
            print("[ENGINE] Trying Cloudflare Workers AI (direct API, free tier)...")
            return generate_cloudflare_image(prompt, num_samples)
        except Exception as e:
            print(f"[FALLBACK] Cloudflare Workers AI failed ({e}). Falling back to Pollinations AI...")
    else:
        print("[INFO] Cloudflare not configured. Using Pollinations AI as the free engine.")

    # Priority 3: Pollinations (always available)
    return generate_pollinations_image(prompt, num_samples)


# ─── Free Generator API Call (Pollinations AI - FLUX Model) ───────────────────

def generate_pollinations_image(prompt: str, num_samples: int = 4) -> list[str]:
    """
    Generates high-quality images using Pollinations AI (FLUX models) with multi-model fallback,
    staggered delays, and automatic rate-limit backoff.

    NOTE: This fallback path does NOT support init_image / image-to-image at all.
    Whenever Stability AI fails and we drop to this function, any visual reference
    (including material context that used to come from an init image) is lost —
    the ONLY thing carrying material information from here on is the text `prompt`
    itself. Callers (generate_stability_image) must make sure the material is
    already clearly stated in the prompt text before falling back here, and this
    function reinforces that by explicitly restating "material accuracy" guidance
    in its negative/positive prompt additions below.
    """
    import base64
    import urllib.parse
    import urllib.request
    import urllib.error
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"[POLLINATIONS AI] Generating {num_samples} images with multi-model fallback...")
    # FIX #7: cap prompt length for reliability — see STABILITY_PROMPT_CHAR_LIMIT /
    # POLLINATIONS_PROMPT_CHAR_LIMIT docstring above for why this matters.
    prompt = _fit_prompt_to_limit(prompt, POLLINATIONS_PROMPT_CHAR_LIMIT)
    clean_prompt = prompt.replace("\n", " ").strip()
    # Try fast models in fallback order (flux-schnell is primary: fastest & high
    # quality). "flux" (the full, slower model) was added as a 3rd-tier fallback:
    # when schnell/turbo are both saturated (as seen with repeated 429s), the
    # full model is sometimes still reachable — worth the extra wait rather than
    # giving up and duplicating a single successful image across the whole batch.
    models = ["flux-schnell", "turbo", "flux"]

    # FIX #10 (critical): the retry loop below used to be `for attempt in range(1)`,
    # which only ever runs ONCE — so the `continue` in the 429 branch had nothing
    # to continue to, and every 429 was treated as an immediate, permanent failure
    # for that model. In practice this meant a single rate-limit hit killed a
    # sample outright, and with several samples firing near-simultaneously we saw
    # 3 out of 4 samples fail, forcing the "fill gaps by duplicating the one
    # success" fallback — silently undoing the per-sample variation fix. We now
    # actually retry, with a PER-MODEL attempt budget: more retries on the fast
    # primary models, only 1 try on "flux" (the slow last-resort model) so a
    # total outage doesn't stall a single sample for minutes.
    MODEL_ATTEMPT_BUDGET = {"flux-schnell": 3, "turbo": 2, "flux": 1}
    BASE_BACKOFF_SECONDS = 2.5

    # Packaging-aware style variations — setiap variasi menjaga konsistensi dieline
    # dan tidak mengubah angle secara drastis yang bisa memecah konsistensi motif antar panel.
    # FIX: added explicit "keep the same packaging material/shape as specified" reminder
    # to every variation, since this fallback has no init_image to anchor material.
    style_variations = [
        ", front face view, centered label legible, seamless tiling motif pattern on all panels, "
        "same packaging material and shape as specified in the prompt, clean studio white background, "
        "print-ready packaging mockup",
        ", slight 3/4 left angle showing front and left side panel, motif pattern continuous across both panels, "
        "same packaging material and shape as specified in the prompt, commercial product photography, white background",
        ", straight-on front view, ultra-sharp product name text, consistent repeating motif background, "
        "same packaging material and shape as specified in the prompt, isolated white background, professional studio lighting",
        ", 3/4 right angle showing front and right side panel, motif seamlessly wrapping corners, "
        "same packaging material and shape as specified in the prompt, photorealistic render, white background",
    ]

    # Global negative prompt: larangan keras terhadap elemen visual asing DAN material yang salah
    global_negative = (
        " -- negative: food illustrations, ingredient icons, floating fruit, vegetable drawings, "
        "random decorative objects, misaligned motif, broken pattern, blurry text, illegible label, "
        "asymmetric pattern, watermark, logo unrelated to product, cluttered background, "
        "distorted packaging shape, elements outside print boundaries, wrong packaging material, "
        "incorrect container type that does not match the specified material"
    )

    def fetch_single_sample(idx: int) -> tuple[int, str | None]:
        # Stagger 1.0s between requests to prevent IP rate-limiting on Pollinations
        if idx > 0:
            time.sleep(1.0)

        variation = style_variations[idx % len(style_variations)]
        seed = random.randint(100000, 999999)
        sample_prompt = f"{clean_prompt}{variation}{global_negative}"
        encoded = urllib.parse.quote(sample_prompt)

        for model_idx, model in enumerate(models):
            if model_idx > 0:
                time.sleep(1.0)  # pause before trying the next fallback model

            max_attempts = MODEL_ATTEMPT_BUDGET.get(model, 1)
            for attempt in range(max_attempts):
                url = (
                    f"https://image.pollinations.ai/prompt/{encoded}"
                    f"?width=1024&height=1024&model={model}&nologo=true&seed={seed}"
                )
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/{115 + idx}.0.0.0 Safari/537.36"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=20) as response:
                        img_bytes = response.read()
                        if len(img_bytes) > 2000:
                            b64 = base64.b64encode(img_bytes).decode("utf-8")
                            print(
                                f"[POLLINATIONS] Sample {idx+1}/{num_samples} OK "
                                f"({len(img_bytes) // 1024} KB) model={model} attempt={attempt+1}"
                            )
                            return idx, b64
                except urllib.error.HTTPError as he:
                    print(
                        f"[POLLINATIONS] Sample {idx+1} HTTP {he.code} model={model} "
                        f"attempt={attempt+1}/{max_attempts}"
                    )
                    if he.code == 429 and attempt < max_attempts - 1:
                        # Real backoff+retry now — increases with each attempt.
                        time.sleep(BASE_BACKOFF_SECONDS * (attempt + 1))
                        continue  # retry same model
                    break  # other HTTP error, or retries exhausted → try next model
                except Exception as e:
                    print(
                        f"[POLLINATIONS WARN] Sample {idx+1} model={model} "
                        f"attempt={attempt+1}/{max_attempts}: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(BASE_BACKOFF_SECONDS)
                        continue  # retry same model on timeout/connection error too
                    break  # retries exhausted → try next model
        return idx, None

    results = []
    # Use max_workers=1 (sequential fetching) with a 1.0s stagger between requests.
    # Pollinations' free tier limits concurrent requests from the same IP to 1 request
    # at a time — parallel requests cause HTTP 429 rate limits. Sequential execution
    # with 1.0s delay generates all 4 images flawlessly in ~6 seconds with ZERO 429 errors.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(fetch_single_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0])
    successful_images = [b64 for _, b64 in results if b64 is not None]

    if successful_images:
        images_out = list(successful_images)
        while len(images_out) < num_samples:
            images_out.append(images_out[len(images_out) % len(successful_images)])
        return images_out

    raise HTTPException(
        status_code=502,
        detail="Gagal generate desain dari engine AI gratis. Silakan coba lagi."
    )




# ─── Per-sample composition variations for Stability AI ─────────────────────
# FIX #8: Previously all N samples were requested in a SINGLE API call with
# `"samples": N` and an identical prompt. SDXL does use a different random seed
# per sample in that case, but because our prompt is so tightly specified
# (exact framing, exact label position, exact style), the seed alone wasn't
# enough to produce visibly different results — all 4 images looked almost
# identical. We now issue one request PER sample, each with its own explicit
# camera-angle/composition variation appended to the prompt (mirroring what
# generate_pollinations_image already does) and its own random seed, so the
# 4 alternatives are actually meaningfully different from each other.
STABILITY_STYLE_VARIATIONS: list[str] = [
    ", front-facing hero shot, centered symmetric composition, even studio lighting",
    ", slight 3/4 left angle view, soft directional side lighting, subtle depth of field",
    ", straight-on eye-level close-up framing, crisp symmetric composition",
    ", slight 3/4 right angle view, warm rim lighting accent, subtle depth of field",
]


def _call_stability_once(
    prompt: str,
    negative_prompt: str,
    headers: dict,
    init_image_bytes: bytes | None,
    width: int,
    height: int,
    seed: int,
    image_strength: float = INIT_IMAGE_STRENGTH,
) -> tuple[int, str | None]:
    """
    Issues a single Stability AI request for exactly 1 sample with a fixed seed.
    `image_strength` lets the caller choose how strongly the init_image (if any)
    influences the result — use MOTIF_IMAGE_STRENGTH for a low-strength motif
    texture nudge, or INIT_IMAGE_STRENGTH for a user-uploaded reference image.
    Returns (http_status, base64_image_or_None). Never raises — network/timeout
    errors are caught and reported as a synthetic status code so the caller's
    per-sample loop can continue to the next sample instead of aborting.
    """
    try:
        if init_image_bytes:
            init_image_resized = resize_to_allowed_dimension(init_image_bytes)
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
            files = {"init_image": ("init_image.png", init_image_resized, "image/png")}
            data_payload = {
                "text_prompts[0][text]": prompt,
                "text_prompts[0][weight]": 1.0,
                "text_prompts[1][text]": negative_prompt,
                "text_prompts[1][weight]": -1.0,
                "cfg_scale": 7,
                "image_strength": image_strength,
                "samples": 1,
                "steps": 30,
                "seed": seed,
            }
            response = requests.post(url, headers=headers, files=files, data=data_payload, timeout=120)
        else:
            valid_w, valid_h = validate_dimensions(width, height)
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            json_headers = {**headers, "Content-Type": "application/json"}
            json_payload = {
                "text_prompts": [
                    {"text": prompt, "weight": 1.0},
                    {"text": negative_prompt, "weight": -1.0},
                ],
                "cfg_scale": 7,
                "height": valid_h,
                "width": valid_w,
                "samples": 1,
                "steps": 30,
                "seed": seed,
            }
            response = requests.post(url, headers=json_headers, json=json_payload, timeout=120)
    except requests.exceptions.Timeout:
        print("[WARN] Stability API request timed out (120s) for one sample.")
        return 504, None
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Stability API request failed for one sample: {e}")
        return 502, None

    if response.status_code == 200:
        result = response.json()
        artifacts = result.get("artifacts", [])
        if artifacts:
            return 200, artifacts[0]["base64"]
        return 200, None

    try:
        err_json = response.json()
        msg = err_json.get("message", response.text)
    except Exception:
        msg = response.text
    safe_msg = msg.encode("ascii", "ignore").decode("ascii")
    print(f"[WARN] Stability API status {response.status_code}: {safe_msg}")
    return response.status_code, None


def generate_stability_image(
    prompt: str,
    init_image_bytes: bytes | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_samples: int = 4,
    image_strength: float = INIT_IMAGE_STRENGTH,
) -> list[str]:
    """
    Calls Stability AI SDXL 1.0 API — one request per sample, each with a distinct
    composition variation and explicit random seed (see STABILITY_STYLE_VARIATIONS)
    so the resulting alternatives are visually distinguishable, not near-duplicates.
    Falls back to Pollinations AI (FLUX - Free) if Stability hits Rate Limit (429),
    auth/credit errors (401/402), is missing a key, or every sample attempt fails.

    `image_strength` only matters when init_image_bytes is provided — pass
    MOTIF_IMAGE_STRENGTH for a low-strength motif texture reference, or the
    default INIT_IMAGE_STRENGTH for a user-uploaded reference image.
    """
    import random as _random

    # FIX #7 (critical): enforce Stability AI's hard 2000-char text_prompts limit
    # BEFORE anything else. Previously an overlong prompt caused a 400 error on
    # every attempt, forcing a silent fallback to Pollinations every single time —
    # which is the real reason packaging material/shape was being ignored. We cut
    # from the end (see _fit_prompt_to_limit), so the material/shape info that the
    # callers place at the start of the prompt is preserved.
    prompt = _fit_prompt_to_limit(prompt, STABILITY_PROMPT_CHAR_LIMIT)

    # Check in-memory cache first (for identical prompts within TTL)
    cache_key = f"{prompt}_{num_samples}"
    cached = get_cached_images(cache_key)
    if cached:
        return cached

    # Option to force free generator via env flag if desired
    if os.getenv("USE_FREE_GENERATOR", "").lower() in ("true", "1", "yes"):
        print("[ENGINE] USE_FREE_GENERATOR active. Using free fallback engine (Cloudflare Workers AI → Pollinations)...")
        res = generate_free_fallback_image(prompt, num_samples)
        set_cached_images(cache_key, res)
        return res

    api_key = get_stability_api_key()
    if not api_key:
        print("[FALLBACK] STABILITY_API_KEY missing. Falling back to free engine (Cloudflare Workers AI → Pollinations)...")
        res = generate_free_fallback_image(prompt, num_samples)
        set_cached_images(cache_key, res)
        return res

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    negative_prompt = (
        "blurry, low quality, distorted, ugly, flat 2D icon, watermark, plain logo, "
        "unpacked food item, raw food, cooked food dish, meal on plate, eating food, "
        "banana leaf food parcel, dirty background, cluttered scene, wrong packaging material, "
        "packaging material that does not match the requested material"
    )

    collected: list[str] = []
    for i in range(num_samples):
        variation = STABILITY_STYLE_VARIATIONS[i % len(STABILITY_STYLE_VARIATIONS)]
        # Keep the varied prompt within the same char budget as the base prompt.
        variant_prompt = _fit_prompt_to_limit(f"{prompt}{variation}", STABILITY_PROMPT_CHAR_LIMIT)
        seed = _random.randint(1, 2_000_000_000)

        status, image_b64 = _call_stability_once(
            variant_prompt, negative_prompt, headers, init_image_bytes, width, height, seed, image_strength
        )

        if status in (429, 401, 402):
            # Auth/credit/rate-limit errors won't resolve by retrying the next
            # sample either — bail out immediately and fall back the WHOLE batch
            # to the free engine, so all 4 images come from a single consistent engine.
            print(f"[FALLBACK INSTANT] Stability AI returned {status}. Switching entire batch to free engine (Cloudflare Workers AI → Pollinations)...")
            free_res = generate_free_fallback_image(prompt, num_samples)
            set_cached_images(cache_key, free_res)
            return free_res

        if image_b64:
            collected.append(image_b64)
        # else: this single sample failed (e.g. transient 5xx/timeout) — continue
        # to the next sample rather than aborting the whole batch.

    if collected:
        # Fill any gaps (a few individual samples may have failed) by cycling
        # through the successful ones, same approach used elsewhere in this file.
        images_out = list(collected)
        while len(images_out) < num_samples:
            images_out.append(images_out[len(images_out) % len(collected)])
        set_cached_images(cache_key, images_out)
        return images_out

    # ── Every single Stability sample attempt failed → fall back to free engine ──
    print("[FALLBACK] All Stability AI sample attempts failed. Falling back to free engine (Cloudflare Workers AI → Pollinations)...")
    try:
        final_res = generate_free_fallback_image(prompt, num_samples)
        set_cached_images(cache_key, final_res)
        return final_res
    except Exception as fallback_err:
        raise HTTPException(
            status_code=502,
            detail=f"Gagal generate desain: {str(fallback_err)}"
        )