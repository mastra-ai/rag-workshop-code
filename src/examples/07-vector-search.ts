import { embed } from "ai";
import { mastra } from "../mastra";
import { openai } from "@ai-sdk/openai";

async function searchExample() {
  // Example 1: Basic search
  const { embedding: basicEmbedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: "What are the main features of vector databases?",
  });

  const pgVector = mastra.getVector("pg");

  const basicResults = await pgVector.query({
    indexName: "searchexamples",
    queryVector: basicEmbedding,
    topK: 3,
  });

  console.log("\nBasic Search Results:");
  console.log(basicResults);

  // Example 2: Search with metadata filter
  const { embedding: filteredEmbedding } = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: "How to implement vector search?",
  });

  const filteredResults = await pgVector.query({
    indexName: "searchexamples",
    queryVector: filteredEmbedding,
    topK: 3,
    filter: {
      section: "implementation",
    },
  });

  console.log("\nFiltered Search Results:");
  console.log(filteredResults);
}

searchExample().catch(console.error);
