import { MDocument } from "@mastra/rag";
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { mastra } from "../mastra";

// Example documents to insert
const documents = [
  {
    text: `# Vector Database Guide

## Introduction
Vector databases are specialized databases that store and index high-dimensional vectors for similarity search. They are essential for modern AI applications, especially in semantic search and recommendation systems.

## Key Features
- Efficient similarity search
- Support for high-dimensional data
- Optimized for machine learning applications
- Scalable for large datasets`,
    metadata: {
      source: "vector-db-guide.md",
      type: "documentation",
      section: "introduction",
    },
  },
  {
    text: `# Search Implementation Guide

## Basic Search
Here's how to implement a basic vector search:

\`\`\`typescript
async function searchVectors(query: string) {
  // Convert query to embedding
  const { embedding } = await embed(query);
  
  // Search the vector store
  const results = await vectorStore.search({
    vector: embedding,
    topK: 5
  });
  
  return results;
}
\`\`\`

## Advanced Features
For better results, consider:
- Using metadata filters
- Implementing reranking
- Adjusting similarity thresholds`,
    metadata: {
      source: "search-guide.md",
      type: "documentation",
      section: "implementation",
    },
  },
];

async function upsertExampleVectors() {
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
  await pgVector.deleteIndex({ indexName: "searchExamples" });
  // Create index
  await pgVector.createIndex({
    indexName: "searchExamples",
    dimension: 1536,
  });
  // Upsert vectors
  await pgVector.upsert({
    indexName: "searchExamples",
    vectors: embeddings,
    metadata: chunks.map((chunk) => ({
      ...chunk.metadata,
      text: chunk.text,
    })),
  });

  console.log(`Successfully upserted ${chunks.length} embeddings`);
}

// Run the example
upsertExampleVectors().catch(console.error);

/*
Example output:
Successfully upserted 2 vectors

The vectors are now stored in PgVector and can be queried using:
- Basic search
- Metadata filtering
- Similarity search
*/
