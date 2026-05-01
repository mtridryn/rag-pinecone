import fs from 'fs';
import dotenv from 'dotenv';
import { Pinecone } from '@pinecone-database/pinecone';
import { pipeline } from '@xenova/transformers';

dotenv.config();
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

if (!PINECONE_API_KEY || !PINECONE_INDEX_NAME) {
  console.error("Error: PINECONE_API_KEY atau PINECONE_INDEX_NAME belum di-set di .env");
  process.exit(1);
}

// Inisialisasi Pinecone
const pc = new Pinecone({
  apiKey: PINECONE_API_KEY
});
const index = pc.index(PINECONE_INDEX_NAME);

async function ingestData() {
  try {
    console.log("Membaca dokumen knowledge_base.txt...");
    const text = fs.readFileSync('knowledge_base.txt', 'utf-8');

    const chunks = text.split('\n').map(c => c.trim()).filter(c => c.length > 0);
    console.log(`Ditemukan ${chunks.length} chunks.`);

    console.log("Memuat model embedding Xenova/all-MiniLM-L6-v2 secara lokal (mungkin butuh waktu saat pertama kali run)...");
    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    console.log("Memulai proses embedding...");

    const upsertData = [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      const output = await extractor(chunk, {
        pooling: 'mean',
        normalize: true,
      });

      const embedding = Array.from(output.data);

      upsertData.push({
        id: `id_${i + 1}`,
        values: embedding,
        metadata: { text: chunk }
      });

      console.log(`Berhasil embed chunk ${i + 1}/${chunks.length}`);
    }

    console.log(`Menyimpan ${upsertData.length} vektor ke dalam index Pinecone (${PINECONE_INDEX_NAME})...`);
    
    // Pinecone SDK requires passing an array. Depending on the version, it might require it wrapped in an object or array.
    // We try passing the array directly, which is the standard v3+ way. If it fails, we wrap it.
    try {
      await index.upsert(upsertData);
    } catch (err) {
      if (err.message.includes("Must pass in at least 1 record")) {
        // Fallback for some SDK versions
        await index.upsert({ records: upsertData });
      } else {
        throw err;
      }
    }

    console.log("\nSukses! Data Pariwisata telah disimpan di dalam server Pinecone.");
    console.log("Data siap digunakan untuk pencarian semantic!");

  } catch (error) {
    console.error("Terjadi kesalahan:", error.message);
  }
}

ingestData();
