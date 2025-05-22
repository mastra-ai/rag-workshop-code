import { MDocument } from "@mastra/rag";
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { mastra } from "../mastra";

// Example documents to insert
const documents = [
  {
    text: "This is a document about cats.",
    metadata: {
      source: "cats.md",
      type: "test",
      section: "animals",
    },
  },
  {
    text: "This is a document about dogs.",
    metadata: {
      source: "dogs.md",
      type: "test",
      section: "animals",
    },
  },
];

async function updateDeleteExampleVectors() {
  // Create MDocument instance
  const doc = new MDocument({
    docs: documents,
    type: "markdown",
  });

  // Chunk the documents
  await doc.chunk({
    strategy: "markdown",
    headers: [
      ["#", "Header 1"],
      ["##", "Header 2"],
    ],
  });

  // Get chunked documents
  const chunks = doc.getDocs();

  // Generate embeddings
  const { embeddings } = await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: chunks.map((chunk) => chunk.text),
  });

  // Get PgVector instance
  const pgVector = mastra.getVector("pg");

  // Delete existing index (if exists)
  await pgVector.deleteIndex({ indexName: "update_delete" });
  // Create index
  await pgVector.createIndex({
    indexName: "update_delete",
    dimension: 1536,
  });
  // Upsert vectors
  await pgVector.upsert({
    indexName: "update_delete",
    vectors: embeddings,
    metadata: chunks.map((chunk) => ({
      ...chunk.metadata,
      text: chunk.text,
    })),
  });

  console.log(`Successfully upserted ${chunks.length} embeddings`);

  // --- Fetch all vectors to get IDs ---
  const dummyEmbedding = (await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: ["cats"],
  })).embeddings[0];
  const allVectors = await pgVector.query({
    indexName: "update_delete",
    queryVector: dummyEmbedding,
    topK: 10,
  });
  console.log('All vectors after upsert:', JSON.stringify(allVectors, null, 2));

  // --- Test updateVector ---
  const catVector = allVectors[0];
  console.log('First vector object:', JSON.stringify(catVector, null, 2));
  const vectorId = catVector.id;
  if (!vectorId) {
    throw new Error('Could not determine vector ID from returned object');
  }
  const updatedText = "This is the UPDATED document about cats. Now with more info!";
  const updatedEmbedding = (await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: [updatedText],
  })).embeddings[0];

  await pgVector.updateVector({
    indexName: "update_delete",
    id: vectorId,
    update: {
      vector: updatedEmbedding,
      metadata: { ...catVector.metadata, text: updatedText },
    },
  });
  const afterUpdate = await pgVector.query({
    indexName: "update_delete",
    queryVector: dummyEmbedding,
    topK: 10,
  });
  console.log('All vectors after update:', JSON.stringify(afterUpdate, null, 2));

  // --- Test deleteVector ---
  await pgVector.deleteVector({
    indexName: "update_delete",
    id: vectorId,
  });
  const afterDelete = await pgVector.query({
    indexName: "update_delete",
    queryVector: dummyEmbedding,
    topK: 10,
  });
  console.log('All vectors after delete:', JSON.stringify(afterDelete, null, 2));

  process.exit(0);
}

// Run the example
updateDeleteExampleVectors().catch(console.error);
