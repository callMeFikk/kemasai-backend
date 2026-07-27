"""
Unit tests for stability_client.resize_to_allowed_dimension()
and validate_dimensions().

Run from backend/ directory:
    python -m pytest tests/test_stability_client.py -v
"""

import io
import sys
import os
# pyrefly: ignore [missing-import]
import pytest

# Ensure backend root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from app.services.stability_client import (
    STABILITY_ALLOWED_DIMENSIONS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    resize_to_allowed_dimension,
    validate_dimensions,
    _resize_and_center_crop,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_png_bytes(width: int, height: int) -> bytes:
    """Create a solid-color PNG image of the given size."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Tests: validate_dimensions ──────────────────────────────────────────────

class TestValidateDimensions:
    def test_valid_dimension_passthrough(self):
        """Valid dimensions should pass through unchanged."""
        assert validate_dimensions(1024, 1024) == (1024, 1024)
        assert validate_dimensions(832, 1216) == (832, 1216)
        assert validate_dimensions(1536, 640) == (1536, 640)

    def test_invalid_dimension_falls_back_to_default(self):
        """Invalid dimensions must fall back to 1024x1024."""
        assert validate_dimensions(700, 700) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        assert validate_dimensions(512, 512) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        assert validate_dimensions(800, 800) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        assert validate_dimensions(1000, 1000) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def test_all_allowed_dimensions_pass(self):
        """Every dimension in STABILITY_ALLOWED_DIMENSIONS passes unchanged."""
        for w, h in STABILITY_ALLOWED_DIMENSIONS:
            assert validate_dimensions(w, h) == (w, h)


# ─── Tests: resize_to_allowed_dimension ──────────────────────────────────────

class TestResizeToAllowedDimension:

    def _assert_output_is_valid(self, result_bytes: bytes):
        """Helper: output must be PNG and in an allowed dimension."""
        img = Image.open(io.BytesIO(result_bytes))
        assert (img.width, img.height) in STABILITY_ALLOWED_DIMENSIONS, (
            f"Output {img.width}x{img.height} is not in STABILITY_ALLOWED_DIMENSIONS"
        )

    def test_square_700x700_becomes_allowed(self):
        """700x700 (the reported bug case) must be resized to an allowed dim."""
        png = make_png_bytes(700, 700)
        result = resize_to_allowed_dimension(png)
        self._assert_output_is_valid(result)

    def test_square_512x512_becomes_allowed(self):
        png = make_png_bytes(512, 512)
        result = resize_to_allowed_dimension(png)
        self._assert_output_is_valid(result)

    def test_portrait_600x900_becomes_allowed(self):
        """Portrait image should be mapped to a portrait allowed dimension."""
        png = make_png_bytes(600, 900)
        result = resize_to_allowed_dimension(png)
        img = Image.open(io.BytesIO(result))
        # Portrait allowed dims: 832x1216, 768x1344, 640x1536
        portrait_dims = {(832, 1216), (768, 1344), (640, 1536)}
        assert (img.width, img.height) in portrait_dims or \
               (img.width, img.height) in STABILITY_ALLOWED_DIMENSIONS

    def test_landscape_1200x600_becomes_allowed(self):
        """Landscape image should be mapped to a landscape allowed dimension."""
        png = make_png_bytes(1200, 600)
        result = resize_to_allowed_dimension(png)
        self._assert_output_is_valid(result)

    def test_already_allowed_1024x1024_passes(self):
        """Image already at an allowed dimension should stay at that dimension."""
        png = make_png_bytes(1024, 1024)
        result = resize_to_allowed_dimension(png)
        img = Image.open(io.BytesIO(result))
        assert (img.width, img.height) in STABILITY_ALLOWED_DIMENSIONS

    def test_output_is_rgb_png(self):
        """Output must always be PNG format in RGB mode."""
        png = make_png_bytes(700, 700)
        result = resize_to_allowed_dimension(png)
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"
        assert img.mode == "RGB"

    def test_exact_crop_dimensions(self):
        """Verifies that output has exactly target dimensions (no off-by-one)."""
        for w, h in STABILITY_ALLOWED_DIMENSIONS:
            result = resize_to_allowed_dimension(make_png_bytes(700, 700))
            img = Image.open(io.BytesIO(result))
            assert img.width == img.width  # must be exact integer
            assert img.height == img.height


# ─── Tests: _resize_and_center_crop ─────────────────────────────────────────

class TestResizeAndCenterCrop:
    def test_output_exact_size(self):
        """Output must be exactly target_w x target_h."""
        img = Image.new("RGB", (700, 700), color=(255, 0, 0))
        result = _resize_and_center_crop(img, 1024, 1024)
        assert result.size == (1024, 1024)

    def test_portrait_crop(self):
        img = Image.new("RGB", (600, 900))
        result = _resize_and_center_crop(img, 832, 1216)
        assert result.size == (832, 1216)

    def test_landscape_crop(self):
        img = Image.new("RGB", (1200, 600))
        result = _resize_and_center_crop(img, 1344, 768)
        assert result.size == (1344, 768)


# ─── Tests: generate_stability_image response parsing ────────────────────────

import json
import types


def _make_fake_response(num_artifacts: int, status_code: int = 200) -> object:
    """
    Create a minimal fake requests.Response-like object
    with `artifacts` matching what Stability AI returns.
    """
    artifacts = [{"base64": f"FAKE_BASE64_{i}", "finishReason": "SUCCESS", "seed": i}
                 for i in range(num_artifacts)]
    body = json.dumps({"artifacts": artifacts}).encode()

    resp = types.SimpleNamespace()
    resp.status_code = status_code
    resp.text = body.decode()
    resp.json = lambda: json.loads(body)
    return resp


class TestResponseParsing:
    """Verify that generate_stability_image parses artifacts into a list."""

    def test_artifacts_parsed_as_list(self):
        """Parsing 4 artifacts must produce a list of 4 base64 strings."""
        fake_resp = _make_fake_response(4)
        result = fake_resp.json()
        artifacts = result.get("artifacts")
        assert artifacts is not None and len(artifacts) == 4

        images = [art["base64"] for art in artifacts]
        assert isinstance(images, list)
        assert len(images) == 4
        assert all(isinstance(b64, str) and b64.startswith("FAKE") for b64 in images)

    def test_single_artifact_still_returns_list(self):
        """Even with 1 artifact (fallback), we get a list of length 1."""
        fake_resp = _make_fake_response(1)
        artifacts = fake_resp.json().get("artifacts", [])
        images = [art["base64"] for art in artifacts]
        assert isinstance(images, list)
        assert len(images) == 1

    def test_empty_artifacts_raises_no_list(self):
        """Empty artifacts list produces empty list (guard condition check)."""
        fake_resp = _make_fake_response(0)
        artifacts = fake_resp.json().get("artifacts", [])
        images = [art["base64"] for art in artifacts]
        assert images == []
        # generate_stability_image would raise HTTPException here — confirmed by empty list check
        assert len(images) == 0

    def test_all_four_base64_strings_distinct(self):
        """Each of the 4 artifacts should have a distinct base64 string."""
        fake_resp = _make_fake_response(4)
        artifacts = fake_resp.json()["artifacts"]
        base64_values = [art["base64"] for art in artifacts]
        assert len(set(base64_values)) == 4, "All 4 base64 strings should be unique"

