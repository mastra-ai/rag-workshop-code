import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { embed } from "ai";
import { openai } from "@ai-sdk/openai";

export const basicSearchTool = createTool({
  id: "basic-search",
  description: "Simple vector search across all documents",
  inputSchema: z.object({
    query: z.string().describe("The search query"),
    limit: z.number().default(5).describe("Number of results to return"),
  }),
  execute: async ({ context: { query, limit }, mastra }) => {
    const vectorStore = mastra?.getVector("pg");
    if (!vectorStore) {
      throw new Error("Vector store not found");
    }

    const { embedding } = await embed({
      model: openai.embedding("text-embedding-3-small"),
      value: query,
    });

    // Simple search with no filters or special processing
    const results = await vectorStore.query({
      indexName: "workshop",
      queryVector: embedding,
      topK: limit,
    });

    return results;
  },
});
