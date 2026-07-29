---
title: Kemasai Backend
emoji: 🎁
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# SulselPak AI Backend

FastAPI backend for SulselPak AI — generates traditional South Sulawesi packaging designs using Stability AI.

## Endpoints

- `GET /` — Status
- `GET /health` — Health check
- `GET /motifs` — Daftar motif Sulsel
- `GET /categories?type=makanan` — Daftar kategori produk
- `GET /materials` — Daftar material kemasan
- `POST /generate` — Generate desain (JSON)
- `POST /api/generate-design` — Generate desain (Form-data)

## Environment Variables

Set `STABILITY_API_KEY` in Space Settings → Secrets.
