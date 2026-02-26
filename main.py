from fastapi import FastAPI
from pydantic import BaseModel
import requests
import base64

app = FastAPI()

STABILITY_API_KEY = "sk-KsJGsTnsMHjL4zMNrFQ8Bn2qQll5XLDYwlgntoHMU7jq9GSz"

class DesignRequest(BaseModel):
    description: str
    category: str
    motif: str
    eco: bool
    image: str | None = None


def build_prompt(data: DesignRequest):
    material = "eco friendly kraft paper packaging" if data.eco else "modern plastic packaging"

    prompt = f"""
    Professional Indonesian food packaging design.
    Category: {data.category}.
    Description: {data.description}.
    Use local motif: {data.motif}.
    Material: {material}.
    Realistic 3D mockup, branding, studio lighting.
    """

    return prompt


@app.post("/generate")
def generate_design(data: DesignRequest):

    prompt = build_prompt(data)

    if data.image:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
            },
            files={
                "init_image": base64.b64decode(data.image)
            },
            data={
                "text_prompts[0][text]": prompt,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30,
            }
        )
    else:
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}",
            },
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            },
        )

    result = response.json()

    if "artifacts" in result:
        return {"image": result["artifacts"][0]["base64"]}

    return {"error": "Failed generating"}
