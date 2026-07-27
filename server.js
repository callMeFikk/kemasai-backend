require('dotenv').config();
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const OpenAI = require('openai');
const fetch = (...args) => import('node-fetch').then(({ default: f }) => f(...args));

// Canvas for text overlay
let createCanvas, loadImage;
try {
  const canvasModule = require('canvas');
  createCanvas = canvasModule.createCanvas;
  loadImage = canvasModule.loadImage;
  console.log('✅ Canvas module loaded — text overlay enabled');
} catch (e) {
  console.warn('⚠️ Canvas module not available — text overlay disabled:', e.message);
}

const app = express();
const PORT = process.env.PORT || 3000;

// Groq untuk analisis teks (gratis)
const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY || 'dummy_key_to_prevent_startup_crash',
  baseURL: 'https://api.groq.com/openai/v1',
});

// ─── Multer ──────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => cb(null, `sketch-${Date.now()}-${file.originalname}`),
});
const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (/image\/(jpeg|jpg|png)/i.test(file.mimetype) || /\.(png|jpe?g)$/i.test(file.originalname))
      return cb(null, true);
    cb(new Error('Only PNG/JPG allowed'));
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

// ─── DATA MAPS ────────────────────────────────────────────────

const PRODUCT_VISUAL_MAP = {
  'Coto Makassar': { visual: 'beef offal soup in ceramic bowl, dark brown broth, garnished fried shallots', color: 'warm brown amber', category: 'traditional soup' },
  'Konro': { visual: 'grilled beef spare ribs, charred dark surface, caramelized glaze', color: 'dark charcoal brown', category: 'grilled meat' },
  'Pallu Basa': { visual: 'beef offal soup, thick dark coconut milk broth, oil droplets floating', color: 'rich dark brown golden', category: 'traditional soup' },
  'Jalangkote': { visual: 'fried triangular pastry dumpling, golden crispy shell', color: 'golden crispy brown', category: 'fried snack' },
  'Keripik Pisang': { visual: 'thin crispy oval banana chips, golden yellow slices', color: 'golden yellow crispy', category: 'crispy chip snack' },
  'Kopi Toraja': { visual: 'dark roasted arabica coffee beans, premium highland coffee', color: 'deep espresso brown cream', category: 'premium coffee' },
  'Es Pisang Ijo': { visual: 'green banana dessert, bright green sticky rice coating, pink coconut sauce', color: 'bright green pink white', category: 'dessert' },
  'Sarabba': { visual: 'spiced ginger palm sugar drink, amber golden liquid', color: 'warm amber golden caramel', category: 'warm beverage' },
  'Markisa Toraja': { visual: 'passion fruit juice, deep orange-purple color, fresh cut passion fruits', color: 'deep orange purple tropical', category: 'fruit juice' },
};

function resolveProductVisual(product, customName = '') {
  if (PRODUCT_VISUAL_MAP[product]) return PRODUCT_VISUAL_MAP[product];
  const name = (customName + ' ' + product).toLowerCase();
  if (name.includes('pisang')) return { visual: 'banana snack golden yellow', color: 'golden yellow', category: 'banana snack' };
  if (name.includes('kopi')) return { visual: 'dark roasted coffee beans aromatic', color: 'dark espresso brown', category: 'coffee' };
  if (name.includes('keripik')) return { visual: 'crispy chip snack golden brown slices', color: 'golden brown crispy', category: 'chip snack' };
  if (name.includes('es ') || name.includes('jus')) return { visual: 'refreshing tropical drink in glass', color: 'vibrant colorful tropical', category: 'drink' };
  return { visual: `${customName || product} South Sulawesi food product`, color: 'warm natural earthy', category: 'food product' };
}

const MATERIAL_SHAPE_MAP = {
  'Daun Lontar': { anchor: 'handwoven lontar palm leaf basket with lid', style: 'traditional rustic artisan natural' },
  'Pelepah Pisang': { anchor: 'banana stem fiber mat wrap tied with raffia', style: 'eco organic rustic natural' },
  'Kertas Daur Ulang': { anchor: 'kraft paper cardboard box with label sticker', style: 'eco sustainable kraft paper cardboard' },
  'Plastik Ramah Lingkungan': { anchor: 'clear stand-up plastic pouch bag with label', style: 'modern eco-friendly transparent pouch' },
  'Kaca/Botol': { anchor: 'clear glass jar with metal screw lid and label', style: 'premium artisan glass jar' },
};

const MOTIF_VISUAL_MAP = {
  "Pa'Londongan (Toraja)": {
    visual_key: "Toraja Pa'Londongan angular diamond zigzag geometric pattern",
    colors: 'bold red, black, white, yellow',
    placement: 'as prominent border frame and label background pattern',
  },
  "Pa'Limbongan (Toraja)": {
    visual_key: "Toraja Pa'Limbongan interlocking angular cross and diamond shapes",
    colors: 'ochre red, black, earth tone brown',
    placement: 'as repeating geometric pattern covering entire label background',
  },
  "Balo Tettong (Bugis)": {
    visual_key: 'Bugis Balo Tettong colorful vertical stripe silk weaving pattern',
    colors: 'gold, green, red, blue, purple vertical stripes',
    placement: 'as bold full vertical stripe band across entire label',
  },
  "Butta Toa (Makassar)": {
    visual_key: 'Makassar Butta Toa flowing floral vine and leaf ornamental motif',
    colors: 'gold and deep red ornamental flourishes',
    placement: 'as decorative border frame and corner ornaments on label',
  },
  "Tapperek Sanrobone": {
    visual_key: 'Makassar Tapperek Sanrobone octagonal woven mat geometric pattern',
    colors: 'pink, turquoise green, cream woven mat tones',
    placement: 'as octagonal geometric frame and woven border accent on label',
  },
  "Pajonga Bantaeng": {
    visual_key: 'Bantaeng Pajonga deer and plant leaf traditional batik pattern',
    colors: 'warm yellow, green leaves, golden deer illustration',
    placement: 'as artistic deer and botanical background motif on label',
  },
  "Tenun Jeneponto": {
    visual_key: 'Jeneponto traditional handwoven textile fabric motif',
    colors: 'beige, earthy brown, natural fiber tones',
    placement: 'as woven textile fabric texture and border accent on label',
  },
  "Parang Gowa": {
    visual_key: 'Gowa Parang diagonal geometric royal batik motif',
    colors: 'dark brown, black, white, gold diagonal geometric lines',
    placement: 'as classic royal diagonal pattern across label background',
  },
};

// ─── PROMPT BUILDER ───────────────────────────────────────────
function buildPrompt(params, pv, pi, mi) {
  const { productName, isHalal, hasBPOM, netWeight, expiryDate, targetMarket } = params;

  const certs = [
    (isHalal === 'true' || isHalal === true) && 'official Halal certification logo',
    (hasBPOM === 'true' || hasBPOM === true) && 'official BPOM certification logo',
  ].filter(Boolean);

  const infoParts = [
    ...certs,
    netWeight && `Net Weight: ${netWeight}`,
    expiryDate && `Exp: ${expiryDate}`,
  ].filter(Boolean);

  const marketStyle = targetMarket === 'Nasional' ? 'sleek modern commercial retail product' : 'authentic artisan premium local product';

  const positive = [
    `commercial product packaging mockup of ${pi.anchor}`,
    `prominent central front label with bold clear typography title "${productName}"`,
    `label decorated with ${mi.visual_key}, ${mi.colors}, ${mi.placement}`,
    `featuring ${pv.visual}`,
    infoParts.length ? `label badge details: ${infoParts.join(', ')}` : '',
    `packaging style: ${pi.style}, ${marketStyle}, South Sulawesi etnik aesthetic, ${pv.color} color scheme`,
    'clean vector typography, high contrast crisp text, studio lighting, isolated white background, 8k sharp focus commercial product photography'
  ].filter(Boolean).join(', ');

  const negative = 'blurry text, distorted typography, illegible writing, low resolution, noisy background, cluttered, dark shadows, out of frame';

  return { positive, negative };
}

// ─── FREE IMAGE GENERATION (POLLINATIONS AI - FLUX MODEL) ──────
async function generateImagePollinations(prompts) {
  // Prompt focused on visual packaging only — text/typography handled by canvas overlay
  const cleanPrompt = (prompts.positive || '').replace(/\n/g, ' ').trim();
  const variations = [
    ', front view studio product photo, clean white background, no text',
    ', 3/4 angle product mockup, studio lighting, white background, no text',
    ', professional product packaging photo, bright lighting, white background, no text',
    ', side perspective product display, studio white background, no text'
  ];

  console.log(`🎨 [Pollinations AI] Generating packaging image with FLUX model...`);

  for (let i = 0; i < variations.length; i++) {
    const seed = Math.floor(Math.random() * 900000) + 10000;
    const fullPrompt = cleanPrompt + variations[i];
    const encoded = encodeURIComponent(fullPrompt);
    const url = `https://image.pollinations.ai/prompt/${encoded}?width=1024&height=1024&model=flux&nologo=true&seed=${seed}`;

    try {
      console.log(`🎨 Fetching sample ${i + 1} from Pollinations AI...`);
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
        },
        signal: AbortSignal.timeout(35_000),
      });

      if (response.ok) {
        const arrayBuffer = await response.arrayBuffer();
        if (arrayBuffer.byteLength > 2000) {
          const base64 = Buffer.from(arrayBuffer).toString('base64');
          console.log(`✅ Pollinations AI image OK — ${(arrayBuffer.byteLength / 1024).toFixed(0)} KB`);
          return { type: 'base64', data: base64 };
        }
      } else {
        console.warn(`⚠️ Pollinations status ${response.status}`);
      }
    } catch (err) {
      console.warn(`⚠️ Pollinations fetch attempt ${i + 1} failed: ${err.message}`);
    }
  }

  throw new Error('Gagal menghasilkan gambar dari Pollinations AI');
}

// ─── TEXT OVERLAY (CANVAS) ───────────────────────────────────────
/**
 * Overlays readable product label text on top of the AI-generated image.
 * Draws a semi-transparent banner at the bottom of the image with:
 * - Product name (large, bold)
 * - Motif & Material info
 * - Optional Halal / BPOM badge
 */
async function overlayTextOnImage(imageBase64, params) {
  if (!createCanvas || !loadImage) {
    console.warn('⚠️ Canvas not available, skipping text overlay');
    return imageBase64;
  }
  try {
    const { productName, motif, material, isHalal, hasBPOM, netWeight } = params;
    const imgBuffer = Buffer.from(imageBase64, 'base64');
    const img = await loadImage(imgBuffer);

    const W = img.width || 1024;
    const H = img.height || 1024;
    const canvas = createCanvas(W, H);
    const ctx = canvas.getContext('2d');

    // Draw original AI image
    ctx.drawImage(img, 0, 0, W, H);

    // ── Label banner background (bottom 30% of image) ──────────
    const bannerH = Math.round(H * 0.30);
    const bannerY = H - bannerH;
    const gradient = ctx.createLinearGradient(0, bannerY, 0, H);
    gradient.addColorStop(0, 'rgba(0,0,0,0)');
    gradient.addColorStop(0.3, 'rgba(0,0,0,0.72)');
    gradient.addColorStop(1, 'rgba(0,0,0,0.90)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, bannerY, W, bannerH);

    // ── Product Name (large, bold) ─────────────────────────────
    const nameFontSize = Math.round(W * 0.072);
    ctx.font = `bold ${nameFontSize}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Text shadow for readability
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 2;

    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(productName, W / 2, bannerY + bannerH * 0.28);

    // ── Sub label (motif & material) ───────────────────────────
    const subFontSize = Math.round(W * 0.033);
    ctx.font = `${subFontSize}px Arial, sans-serif`;
    ctx.fillStyle = '#F0D58C';
    ctx.shadowBlur = 4;
    const subText = [motif, material].filter(Boolean).join('  ·  ');
    ctx.fillText(subText, W / 2, bannerY + bannerH * 0.55);

    // ── Badges (Halal / BPOM / Netto) ─────────────────────────
    const badges = [];
    if (isHalal === 'true' || isHalal === true) badges.push('🥩 HALAL');
    if (hasBPOM === 'true' || hasBPOM === true) badges.push('🏛 BPOM');
    if (netWeight) badges.push(`⚖️ ${netWeight}`);
    if (badges.length > 0) {
      const badgeFontSize = Math.round(W * 0.026);
      ctx.font = `bold ${badgeFontSize}px Arial, sans-serif`;
      ctx.fillStyle = '#FFFFFF';
      ctx.shadowBlur = 4;
      ctx.fillText(badges.join('   '), W / 2, bannerY + bannerH * 0.80);
    }

    // ── Thin golden divider line ───────────────────────────────
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.strokeStyle = '#C9A84C';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W * 0.1, bannerY + bannerH * 0.42);
    ctx.lineTo(W * 0.9, bannerY + bannerH * 0.42);
    ctx.stroke();

    const finalBase64 = canvas.toBuffer('image/jpeg', { quality: 0.90 }).toString('base64');
    console.log(`✅ Text overlay applied — product: "${productName}"`);
    return finalBase64;
  } catch (err) {
    console.warn('⚠️ Text overlay failed, returning original image:', err.message);
    return imageBase64;
  }
}

// ─── IMAGE GENERATION ROUTER (POLLINATIONS AI FLUX) ─────────────
async function generateImageHuggingFace(prompts) {
  // Use Pollinations AI (FLUX model) directly for fast, free, and reliable 3D packaging renders
  return await generateImagePollinations(prompts);
}

// ─── JSON PARSER HELPER ───────────────────────────────────────
function safeParseJson(rawText) {
  if (!rawText) throw new Error('Raw text is empty');
  let cleanText = rawText.trim();

  if (cleanText.startsWith('```')) {
    cleanText = cleanText.replace(/^```json\s*/i, '').replace(/```$/, '').trim();
  }

  const firstBrace = cleanText.indexOf('{');
  const lastBrace = cleanText.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1) {
    cleanText = cleanText.substring(firstBrace, lastBrace + 1);
  }

  return JSON.parse(cleanText);
}

// ─── PROVIDER: GROQ ───────────────────────────────────────────
async function callGroq(prompt) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY is not configured');

  const models = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama3-8b-8192'];
  for (const model of models) {
    try {
      console.log(`[INFO] Trying Groq model: ${model}`);
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 512,
          temperature: 0.3,
          response_format: { type: 'json_object' }
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        console.warn(`⚠️ Groq model ${model} failed: ${data.error?.message || response.statusText}`);
        continue;
      }
      return data.choices[0].message.content.trim();
    } catch (err) {
      console.warn(`⚠️ Groq model ${model} exception: ${err.message}`);
    }
  }
  throw new Error('All Groq models failed');
}

// ─── PROVIDER: GEMINI ─────────────────────────────────────────
async function callGemini(prompt) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not configured');

  const models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash-latest', 'gemini-1.5-pro'];
  for (const model of models) {
    try {
      console.log(`[INFO] Trying Gemini model: ${model}`);
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.3,
            maxOutputTokens: 1024,
            responseMimeType: 'application/json'
          }
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        console.warn(`⚠️ Gemini model ${model} failed: ${data.error?.message || response.statusText}`);
        continue;
      }
      return data.candidates[0].content.parts[0].text.trim();
    } catch (err) {
      console.warn(`⚠️ Gemini model ${model} exception: ${err.message}`);
    }
  }
  throw new Error('All Gemini models failed');
}

// ─── PROVIDER: OPENROUTER ─────────────────────────────────────
async function callOpenRouter(prompt) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY is not configured');

  const models = ['google/gemini-2.0-flash-lite:free', 'meta-llama/llama-3-8b-instruct:free'];
  for (const model of models) {
    try {
      console.log(`[INFO] Trying OpenRouter model: ${model}`);
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://kemasai.app',
          'X-Title': 'KemasAI App'
        },
        body: JSON.stringify({
          model,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 512,
          temperature: 0.3
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        console.warn(`⚠️ OpenRouter model ${model} failed: ${data.error?.message || response.statusText}`);
        continue;
      }
      return data.choices[0].message.content.trim();
    } catch (err) {
      console.warn(`⚠️ OpenRouter model ${model} exception: ${err.message}`);
    }
  }
  throw new Error('All OpenRouter models failed');
}

// ─── PROVIDER: LOCAL FALLBACK ─────────────────────────────────
function getLocalFallback(params, pv, pi, mi) {
  const { productName, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate } = params;
  return {
    color_palette: ['#1A3A52', '#D4AF37', '#8B4513'],
    typography: 'Serif bold untuk nama brand agar menonjol, dan Sans-serif clean untuk deskripsi produk',
    layout: `Desain terpusat pada kemasan ${material}. Nama produk "${productName}" diletakkan dominan di tengah dengan latar motif ${motif}.`,
    cultural_tips: `Integrasikan motif khas Sulawesi Selatan ${motif} sebagai pemanis visual atau bingkai label produk untuk menonjolkan aspek kearifan lokal.`,
    market_positioning: `Target pasar ${targetMarket}: Menampilkan identitas premium bernuansa budaya lokal yang khas dan ramah lingkungan.`,
    umkm_compliance: `Kesesuaian Regulasi: ${isHalal === 'true' || isHalal === true ? 'Logo Halal terintegrasi.' : 'Disarankan mendaftarkan sertifikasi Halal.'} ${hasBPOM === 'true' || hasBPOM === true ? 'Logo BPOM terintegrasi.' : 'Disarankan pengurusan izin BPOM.'} Pastikan info Berat Bersih (${netWeight || '-'}) dan Kedaluwarsa (${expiryDate || '-'}) tercetak jelas pada label kemasan.`,
    packaging_advantage: `Material kemasan ${material} memberikan perlindungan optimal yang ramah lingkungan dan aman bagi produk pangan.`
  };
}

// ─── AI ROUTER ORCHESTRATOR ───────────────────────────────────
async function getAIAnalysis(params, pv, pi, mi) {
  const { productName, product, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate } = params;

  const prompt = `Kamu konsultan desain kemasan UMKM Sulawesi Selatan.
Produk: ${productName} (${product} — ${pv.category}) | Motif: ${motif} | Material: ${material}
Target: ${targetMarket} | Halal: ${isHalal === 'true' || isHalal === true ? 'Ya' : 'Tidak'} | BPOM: ${hasBPOM === 'true' || hasBPOM === true ? 'Ya' : 'Tidak'}
Berat Bersih: ${netWeight || 'Tidak ditentukan'} | Tanggal Kedaluwarsa: ${expiryDate || 'Tidak ditentukan'}

Berikan analisis dalam format JSON valid dengan field berikut:
{
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "typography": "saran tipografi...",
  "layout": "saran komposisi layout...",
  "cultural_tips": "saran nilai budaya...",
  "market_positioning": "strategi positioning pasar...",
  "umkm_compliance": "saran kesesuaian regulasi UMKM...",
  "packaging_advantage": "keunggulan material kemasan..."
}

Balas HANYA dengan JSON valid. Jangan berikan teks penjelasan lain di luar JSON.`;

  const providers = [
    { name: 'groq', fn: () => callGroq(prompt) },
    { name: 'gemini', fn: () => callGemini(prompt) },
    { name: 'openrouter', fn: () => callOpenRouter(prompt) }
  ];

  for (const provider of providers) {
    try {
      console.log(`[INFO] Router: Attempting ${provider.name}...`);
      const startTime = Date.now();
      const rawText = await provider.fn();
      const analysis = safeParseJson(rawText);
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`[INFO] Router: ${provider.name} succeeded in ${duration}s`);
      return { provider: provider.name, analysis, duration: `${duration}s` };
    } catch (err) {
      console.warn(`⚠️ [WARNING] Provider ${provider.name} failed: ${err.message}`);
    }
  }

  console.log(`[INFO] Router: All online providers failed. Falling back to local rules.`);
  return {
    provider: 'local_fallback',
    analysis: getLocalFallback(params, pv, pi, mi),
    duration: '0s'
  };
}

// ─── ROUTES ───────────────────────────────────────────────────
app.get('/health', (req, res) => {
  const hfToken = process.env.HF_TOKEN || process.env.HF_API_KEY;
  const groqToken = process.env.GROQ_API_KEY;
  const geminiToken = process.env.GEMINI_API_KEY;

  res.json({
    status: 'OK',
    version: '4.0-router',
    message: 'KemasAI Multi-Provider Backend is running!',
    keys: {
      huggingface: hfToken ? '✅ Set' : '❌ Missing (set HF_TOKEN atau HF_API_KEY)',
      groq: groqToken ? '✅ Set' : '❌ Missing (set GROQ_API_KEY)',
      gemini: geminiToken ? '✅ Set' : '❌ Missing (set GEMINI_API_KEY)',
    },
  });
});

app.post('/api/generate-design', upload.single('sketch'), async (req, res) => {
  const sketchFilePath = req.file?.path || null;
  const requestStartTime = Date.now();

  try {
    console.log('\n📝 === NEW REQUEST ===');
    const {
      productName, category, product, motif, material,
      targetMarket, isHalal, hasBPOM, netWeight, expiryDate,
    } = req.body;

    if (!productName)
      return res.status(400).json({ success: false, error: 'productName is required' });


    console.log(`📦 ${productName} | Motif: ${motif} | Material: ${material}`);

    const pv = resolveProductVisual(product, productName);
    const pi = MATERIAL_SHAPE_MAP[material] || { anchor: 'food packaging container with label', style: 'clean product packaging' };
    const mi = MOTIF_VISUAL_MAP[motif] || {
      visual_key: `${motif} traditional South Sulawesi geometric pattern`,
      colors: 'traditional ethnic colorful',
      placement: 'as decorative border on label',
    };

    const prompts = buildPrompt(
      { productName, product, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate },
      pv, pi, mi
    );

    console.log(`📋 Positive Prompt: ${prompts.positive}`);

    console.log('\n🚀 Executing AI Router (text) + HF Image generation in parallel...');
    const [imageResult, analysisResult] = await Promise.allSettled([
      generateImageHuggingFace(prompts),
      getAIAnalysis(
        { productName, product, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate },
        pv, pi, mi
      )
    ]);

    let imageBase64 = null, imageUrl = null;
    if (imageResult.status === 'fulfilled') {
      const d = imageResult.value;
      if (d.type === 'base64') {
        // Apply text overlay so product name is always clearly readable
        imageBase64 = await overlayTextOnImage(d.data, {
          productName, motif, material, isHalal, hasBPOM, netWeight
        });
      } else {
        imageUrl = d.data;
      }
      console.log('✅ Image generation completed successfully');
    } else {
      console.error('❌ Image generation failed:', imageResult.reason?.message);
    }

    let routerResult = { provider: 'local_fallback', analysis: null, duration: '0s' };
    if (analysisResult.status === 'fulfilled') {
      routerResult = analysisResult.value;
    } else {
      console.error('❌ AI Router failed to resolve:', analysisResult.reason?.message);
      routerResult.analysis = getLocalFallback(
        { productName, product, motif, material, targetMarket, isHalal, hasBPOM, netWeight, expiryDate },
        pv, pi, mi
      );
    }

    if (sketchFilePath) {
      try { fs.unlinkSync(sketchFilePath); } catch (_) { }
    }

    const totalDuration = ((Date.now() - requestStartTime) / 1000).toFixed(2);
    console.log(`[INFO] Request processed in ${totalDuration}s. Provider used: ${routerResult.provider}`);

    res.json({
      success: true,
      image: imageBase64 || imageUrl,
      image_base64: imageBase64 || imageUrl, // Backward compatibility
      image_url: imageUrl,
      analysis: routerResult.analysis,
      provider: routerResult.provider,
      timestamp: new Date().toISOString(),
      processing_time: `${totalDuration}s`,
      prompt_used: prompts.positive,
      metadata: {
        version: '4.0-router',
        llm: routerResult.provider,
        image_model: 'huggingface/FLUX.1-schnell',
        mode: 'generate',
        has_sketch: !!req.file,
        has_image: !!(imageBase64 || imageUrl),
        packaging_anchor: pi.anchor,
        product_category: pv.category,
      },
    });

  } catch (error) {
    console.error('❌ Fatal error in endpoint:', error.message);
    if (sketchFilePath) {
      try { fs.unlinkSync(sketchFilePath); } catch (_) { }
    }
    res.status(500).json({ success: false, error: error.message });
  }
});

// ─── ENDPOINT: SIMPAN EVALUASI ───────────────────────────────────
app.post('/api/evaluation', (req, res) => {
  try {
    const { product_name, product_type, scores, feedback, timestamp } = req.body;

    if (!product_name || !scores) {
      return res.status(400).json({ success: false, error: 'product_name and scores are required' });
    }

    // Hitung rata-rata skor
    const scoreValues = Object.values(scores).filter(v => typeof v === 'number');
    const average = scoreValues.length > 0
      ? (scoreValues.reduce((a, b) => a + b, 0) / scoreValues.length).toFixed(2)
      : '0.00';

    const evaluationRecord = {
      id: `eval_${Date.now()}`,
      timestamp: timestamp || new Date().toISOString(),
      product_name,
      product_type: product_type || '',
      scores: { ...scores, average: parseFloat(average) },
      feedback: feedback || {},
    };

    // Log ke console (bisa diganti database di production)
    console.log('[EVALUATION]', JSON.stringify(evaluationRecord));

    res.json({
      success: true,
      message: 'Evaluasi berhasil disimpan',
      evaluation_id: evaluationRecord.id,
      average_score: parseFloat(average),
    });
  } catch (error) {
    console.error('Error saving evaluation:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// ─── ENVIRONMENT VALIDATION ────────────────────────────────────
function validateEnvironment() {
  const hfToken = process.env.HF_TOKEN || process.env.HF_API_KEY;
  const groqToken = process.env.GROQ_API_KEY;
  const geminiToken = process.env.GEMINI_API_KEY;

  console.log('\n🛡️  === STARTUP ENVIRONMENT VALIDATION ===');
  console.log(`   HF_API_KEY/HF_TOKEN : ${hfToken ? '✅ DETECTED' : '⚠️  MISSING (Image generation will fail)'}`);
  console.log(`   GROQ_API_KEY        : ${groqToken ? '✅ DETECTED' : '⚠️  MISSING (Groq will be bypassed)'}`);
  console.log(`   GEMINI_API_KEY      : ${geminiToken ? '✅ DETECTED' : '⚠️  MISSING (Gemini will be bypassed)'}`);
  console.log('   =========================================\n');
}

// ─── START ────────────────────────────────────────────────────
if (!fs.existsSync('./uploads')) fs.mkdirSync('./uploads');

app.listen(PORT, () => {
  validateEnvironment();
  console.log(`🚀 KemasAI Backend v4.0-router — Running on http://localhost:${PORT}\n`);
});
