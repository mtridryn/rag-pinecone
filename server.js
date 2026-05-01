import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Groq from 'groq-sdk';
import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline } from '@xenova/transformers';

dotenv.config();
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

if (!PINECONE_API_KEY || !PINECONE_INDEX_NAME || !GROQ_API_KEY) {
  console.error("Error: Pastikan PINECONE_API_KEY, PINECONE_INDEX_NAME, dan GROQ_API_KEY sudah di-set di .env");
  process.exit(1);
}

// Inisialisasi Klien
const groq = new Groq({ apiKey: GROQ_API_KEY });
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX_NAME);

// Setup Express
const app = express();
const PORT = process.env.PORT || 3001; 

app.use(cors());
app.use(express.json());

// Inisialisasi Model Xenova (hanya sekali saat server menyala)
let extractor;
console.log("Memuat model embedding Xenova/all-MiniLM-L6-v2...");
pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2').then(ext => {
  extractor = ext;
  console.log("Model embedding siap digunakan.");
}).catch(err => {
  console.error("Gagal memuat model embedding:", err);
});


app.post('/api/chat', async (req, res) => {
  try {
    const userMessage = req.body.message;
    if (!userMessage) return res.status(400).json({ error: "Pesan kosong" });

    if (!extractor) {
      return res.status(503).json({ error: "Model AI masih loading, coba lagi dalam beberapa detik..." });
    }

    // 1. Embed Query dengan Xenova
    const output = await extractor(userMessage, {
      pooling: 'mean',
      normalize: true,
    });
    const userVector = Array.from(output.data);

    // 2. Similarity Search menggunakan Pinecone
    const results = await index.query({
      topK: 3,
      vector: userVector,
      includeMetadata: true
    });

    // Ambil dokumen hasil pencarian Pinecone
    console.log("Skor kecocokan Pinecone:", results.matches.map(m => m.score));
    
    const relevantDocs = results.matches
      .filter(match => match.metadata && match.metadata.text)
      .map(match => match.metadata.text);
      
    const bestMatchesText = relevantDocs;
    const contextText = relevantDocs.length > 0
      ? relevantDocs.map(doc => `- ${doc}`).join('\n')
      : '';

    // 3. Generation menggunakan Groq
    const isContextEmpty = contextText === '';
    const systemInstruction = `You are a professional Customer Service Assistant for Mutiara Travel (a travel agency based in Ubud, Bali).

STRICT RULES:
1. LANGUAGE MATCHING (CRITICAL): You MUST detect the language of the user's input and reply 100% in THAT language. If the user speaks English, reply in English. If Russian, reply in Russian. If Chinese, reply in Chinese. DO NOT reply in Indonesian unless the user speaks Indonesian. You MUST translate the facts from the [Context] into the user's language.
2. FORMATTING: Use bullet points, bold text for prices and features, and short paragraphs. DO NOT write long walls of text.
3. COMPLETENESS: Extract and provide comprehensive details (Price, Duration, Facilities, Itinerary) from the [Context] if available.
4. GREETINGS: If the user only says a greeting (e.g., "Hello"), greet them back warmly in their language. If the user asks a specific question, answer it DIRECTLY without a long opening greeting.
5. NO HALLUCINATION: You may ONLY use facts provided in the [Context] below. DO NOT invent packages, prices, or locations.
6. IF UNKNOWN / EMPTY CONTEXT: If the requested information is not in the [Context], you MUST politely decline and provide the email cs@mutiaratravel.co.id. This rejection MUST be in the user's language.

[Context]
${isContextEmpty ? 'EMPTY. Warning: Decline the request politely in the user\'s language and provide the CS email. Do not invent answers.' : contextText}`;

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: 'system', content: systemInstruction },
        { role: 'user', content: userMessage }
      ],
      model: 'llama-3.1-8b-instant',
      temperature: 0.5,
    });

    const chatResponse = chatCompletion.choices[0]?.message?.content || "";

    res.json({ answer: chatResponse, context: bestMatchesText });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Oh no! Terjadi masalah: ${error.message}` });
  }
});

app.listen(PORT, () => {
    console.log(`RAG Pinecone & Groq berjalan di port ${PORT}`);
});
