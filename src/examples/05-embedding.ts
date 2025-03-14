import { embed, embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { cohere } from "@ai-sdk/cohere";

async function embeddingExample() {
  const text = "The quick brown fox jumps over the lazy dog";

  // Example 1: Basic embedding with text-embedding-3-small
  console.log("\n1. OpenAI text-embedding-3-small (Fastest, 1536 dimensions):");
  const { embedding: smallEmbedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: text,
  });
  console.log("Dimensions:", smallEmbedding.length);
  console.log("First 5 values:", smallEmbedding.slice(0, 5));

  // Example 2: Higher quality embedding with text-embedding-3-large
  console.log(
    "\n2. OpenAI text-embedding-3-large (Best quality, 3072 dimensions):"
  );
  const { embedding: largeEmbedding } = await embed({
    model: openai.embedding("text-embedding-3-large"),
    value: text,
  });
  console.log("Dimensions:", largeEmbedding.length);
  console.log("First 5 values:", largeEmbedding.slice(0, 5));

  // Example 3: Cohere embedding model
  console.log("\n3. Cohere embed-english-v3.0 (Multilingual support):");
  const { embedding: cohereEmbedding } = await embed({
    model: cohere.embedding("embed-english-v3.0"),
    value: text,
  });
  console.log("Dimensions:", cohereEmbedding.length);
  console.log("First 5 values:", cohereEmbedding.slice(0, 5));

  // Example 4: Batch embedding multiple texts
  console.log("\n4. Batch embedding multiple texts:");
  const texts = [
    "The quick brown fox",
    "jumps over the lazy dog",
    "and then runs away",
  ];

  const { embeddings: batchEmbeddings } = await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: texts,
  });

  console.log("Number of embeddings:", batchEmbeddings.length);
  console.log("Each embedding dimensions:", batchEmbeddings[0].length);
}

embeddingExample().catch(console.error);

/* Example output:
1. OpenAI text-embedding-3-small (Fastest, 1536 dimensions):
Dimensions: 1536
First 5 values: [0.123, -0.456, 0.789, -0.012, 0.345]

2. OpenAI text-embedding-3-large (Best quality, 3072 dimensions):
Dimensions: 3072
First 5 values: [0.234, -0.567, 0.890, -0.123, 0.456]

3. Cohere embed-english-v3.0 (Multilingual support):
Dimensions: 1024
First 5 values: [0.345, -0.678, 0.901, -0.234, 0.567]

4. Batch embedding multiple texts:
Number of embeddings: 3
Each embedding dimensions: 1536
*/
