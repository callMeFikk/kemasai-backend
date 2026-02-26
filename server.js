// backend/server.js
// Backend KemasAI - Menggunakan Gemini (Google) + Hugging Face (GRATIS)

require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));

const app = express();
const PORT = process.env.PORT || 3000;

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const HF_API_KEY = process.env.HF_API_KEY;

// ✅ FIX: Model lebih ringan & stabil
const HF_MODEL = 'black-forest-labs/FLUX.1-schnell';

const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, `sketch-${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedMime = /image\/(jpeg|jpg|png)/i;
    const allowedExt = /\.(png|jpe?g)$/i;
    if (allowedMime.test(file.mimetype) || allowedExt.test(file.originalname)) {
      return cb(null, true);
    }
    cb(new Error('Only .png, .jpg and .jpeg format allowed!'));
  },
});

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

// ✅ FIX: Gemini dengan fallback model
async function callGemini(prompt, imageBase64 = null) {
  const models = ['gemini-2.0-flash-lite', 'gemini-2.0-flash'];
  
  for (const model of models) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
      const parts = [];

      if (imageBase64 && model !== 'gemini-1.0-pro') {
        parts.push({ inline_data: { mime_type: 'image/jpeg', data: imageBase64 } });
      }
      parts.push({ text: prompt });

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ parts }], generationConfig: { temperature: 0.7, maxOutputTokens: 1024 } }),
      });

      const data = await response.json();
      if (!response.ok) { console.warn(`⚠️ ${model} failed:`, data.error?.message); continue; }
      console.log(`✅ Gemini model used: ${model}`);
      return data.candidates[0].content.parts[0].text;
    } catch (err) {
      console.warn(`⚠️ ${model} error:`, err.message);
    }
  }
  throw new Error('Semua model Gemini gagal');
}

// ✅ FIX: URL baru Hugging Face router
async function generateImageHuggingFace(prompt) {
  console.log('🎨 Generating image via Hugging Face...');

  const response = await fetch(
    `https://router.huggingface.co/hf-inference/models/${HF_MODEL}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          num_inference_steps: 4,
        },
        options: { wait_for_model: true },
      }),
    }
  );

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Hugging Face error: ${errText}`);
  }

  const imageBuffer = await response.buffer();
  return imageBuffer.toString('base64');
}

app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    message: 'KemasAI Backend is running!',
    models: { llm: 'gemini-1.5-flash (FREE)', image: `${HF_MODEL} (FREE)` },
  });
});

app.post('/api/generate-design', upload.single('sketch'), async (req, res) => {
  try {
    console.log('\n📝 === NEW REQUEST ===');
    const { productName, category, product, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate } = req.body;

    if (!productName) return res.status(400).json({ success: false, error: 'productName is required' });

    let sketchBase64 = null;
    if (req.file) {
      sketchBase64 = fs.readFileSync(req.file.path).toString('base64');
      console.log(`📸 Sketch loaded`);
    }

    // Step 1: Gemini Analysis
    console.log('🧠 Step 1: Gemini analysis...');
    const analysisPrompt = `Kamu adalah ahli desain kemasan UMKM Sulawesi Selatan. Analisis desain kemasan:
- Produk: ${productName} (${product})
- Motif: ${motif}, Material: ${material}, Target: ${targetMarket}
- Halal: ${isHalal === 'true' ? 'Ya' : 'Tidak'}, BPOM: ${hasBPOM === 'true' ? 'Ya' : 'Tidak'}

Output HANYA JSON:
{
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "typography": "rekomendasi font",
  "layout": "saran komposisi",
  "cultural_tips": "cara integrasi ${motif}",
  "market_positioning": "strategi branding",
  "image_prompt_en": "professional food packaging design for ${productName}, ${product}, South Sulawesi Indonesia, ${motif} cultural motif, ${material}, vibrant colors, professional label, white background, high quality"
}`;

    let analysis = null;
    let imagePrompt = `professional food packaging design for ${productName}, ${product}, South Sulawesi Indonesia, ${motif} cultural motif, vibrant colors, professional label, white background`;

    try {
      const geminiResponse = await callGemini(analysisPrompt, sketchBase64);
      const jsonMatch = geminiResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        analysis = JSON.parse(jsonMatch[0]);
        imagePrompt = analysis.image_prompt_en || imagePrompt;
        console.log('✅ Gemini OK');
      }
    } catch (err) {
      console.warn('⚠️ Gemini failed, using fallback prompt');
    }

    // Step 2: Hugging Face Image
    console.log('🎨 Step 2: Generating image...');
    let base64Image = null;
    try {
      base64Image = await generateImageHuggingFace(imagePrompt);
      console.log('✅ Image generated!');
    } catch (imgErr) {
      console.error('❌ Image failed:', imgErr.message);
    }

    if (req.file) { try { fs.unlinkSync(req.file.path); } catch (e) {} }

    res.json({
      success: true,
      image_base64: base64Image,
      analysis,
      prompt_used: imagePrompt,
      metadata: { llm: 'gemini-1.5-flash (FREE)', image: HF_MODEL + ' (FREE)', has_image: base64Image !== null },
    });

  } catch (error) {
    console.error('❌ Error:', error.message);
    if (req.file) { try { fs.unlinkSync(req.file.path); } catch (e) {} }
    res.status(500).json({ success: false, error: error.message });
  }
});

if (!fs.existsSync('./uploads')) fs.mkdirSync('./uploads');

app.listen(PORT, () => {
  console.log(`\n🚀 KemasAI Backend (FREE TIER) running on http://localhost:${PORT}`);
  console.log(`📡 Health Check: http://localhost:${PORT}/health`);
  console.log(`🧠 LLM: Gemini 1.5 Flash (FREE)`);
  console.log(`🎨 Image: ${HF_MODEL} (Hugging Face - FREE)`);
  console.log(`\n⚙️  API Keys loaded:`);
  console.log(`   GEMINI: ${GEMINI_API_KEY ? '✅ Set' : '❌ Missing'}`);
  console.log(`   HF: ${HF_API_KEY ? '✅ Set' : '❌ Missing'}\n`);
});