import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { embed } from "ai";
import { openai } from "@ai-sdk/openai";

export const findCodeTool = createTool({
  id: "find-code",
  description: "Finds code snippets, functions, or examples in the codebase",
  inputSchema: z.object({
    query: z
      .string()
      .describe(
        "What to search for (e.g., 'authentication', 'error handling', 'logging')"
      ),
    type: z
      .enum(["function", "method", "class"])
      .default("function")
      .describe("Type of code to find (function, method, or class)"),
    section: z
      .enum(["authentication", "error-handling", "logging"])
      .optional()
      .describe("Section of the codebase to search in"),
  }),
  execute: async ({ context: { query, type, section }, mastra }) => {
    const vectorStore = mastra?.getVector("pg");
    if (!vectorStore) {
      throw new Error("Vector store not found");
    }

    // Enhance query based on type
    const enhancedQuery =
      type === "function"
        ? `function ${query}`
        : type === "method"
          ? `method ${query}`
          : `class ${query}`;

    const { embedding } = await embed({
      model: openai.embedding("text-embedding-3-small"),
      value: enhancedQuery,
    });

    // Build the filter
    const filter: Record<string, any> = {
      type: "code",
      format: "typescript",
    };

    // Add section filter if provided
    if (section) {
      filter.section = section;
    }

    // Add chunk type filter based on the type of code we're looking for
    if (type === "function") {
      filter.chunkType = "function_definition";
    } else if (type === "method") {
      filter.chunkType = "method_definition";
    } else if (type === "class") {
      filter.chunkType = "class_definition";
    }

    const results = await vectorStore.query({
      indexName: "bonus",
      queryVector: embedding,
      topK: 5,
      filter: filter,
    });

    return results;
  },
});
