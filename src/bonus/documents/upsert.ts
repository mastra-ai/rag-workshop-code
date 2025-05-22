import { MDocument } from "@mastra/rag";
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { mastra } from "../../mastra";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';

// Get the directory name of the current file
// This works in both CommonJS and ES modules
const currentFilePath = fileURLToPath(import.meta.url);
const currentDir = path.dirname(currentFilePath);

// Go up to the project root (2 levels up from src/bonus/documents)
const projectRoot = path.resolve(currentDir, '../../..');


/**
 * Upsert code documents to the vector store
 * This function reads code files and indexes them with appropriate metadata
 */
async function upsertCodeDocuments() {
  console.log("Upserting code documents to vector store...");

  // Use the absolute path to the documents directory
  const docsDir = path.join(projectRoot, "src", "bonus", "documents");

  const pgVector = mastra.getVector("pg");
  if (!pgVector) {
    throw new Error("Vector store not found");
  }

  // Define the document files to process
  const documentFiles = [
    {
      path: path.join(docsDir, "authentication-service.ts"),
      type: "code",
      section: "authentication",
      format: "typescript",
    },
    {
      path: path.join(docsDir, "error-handling.ts"),
      type: "code",
      section: "error-handling",
      format: "typescript",
    },
    {
      path: path.join(docsDir, "logger.ts"),
      type: "code",
      section: "logging",
      format: "typescript",
    },
  ];

  // Read and process code files
  const codeContents = await Promise.all(
    documentFiles.map(async (doc) => ({
      text: await fs.readFile(doc.path, "utf-8"),
      metadata: {
        source: path.basename(doc.path),
        type: doc.type,
        section: doc.section,
        format: doc.format,
      },
    }))
  );

  // Process code documents using the MDocument approach
  const codeDoc = new MDocument({
    docs: codeContents,
    type: "text",
  });

  // Chunk the code files using recursive strategy which is better for code
  await codeDoc.chunk({
    strategy: "recursive", // Best for code as it respects code structure
    size: 1500, // Large enough for complete function/class definitions
    overlap: 50, // Small overlap to maintain context between chunks
  });

  // Get the chunked documents
  const codeChunks = codeDoc.getDocs();
  console.log(`Generated ${codeChunks.length} code chunks`);

  // Generate embeddings for all chunks
  const { embeddings } = await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: codeChunks.map((chunk) => chunk.text),
  });

  // Delete existing index (if exists)
  await pgVector.deleteIndex({ indexName: "bonus" });
  await pgVector.createIndex({
    indexName: "bonus",
    dimension: 1536,
  });

  // Upsert the embeddings to the vector store
  await pgVector.upsert({
    indexName: "bonus",
    vectors: embeddings,
    metadata: codeChunks.map((chunk) => ({
      ...chunk.metadata,
      // Add additional metadata for filtering in the tool
      chunkType: getChunkType(chunk.text),
      functionName: getFunctionName(chunk.text, chunk.metadata.source),
      text: chunk.text,
    })),
  });

  console.log("Code documents upserted successfully!");
}

/**
 * Determine the chunk type based on content analysis
 * @param text - The chunk text content
 * @returns The type of code chunk (class, method, function, or file)
 */
function getChunkType(text: string): string {
  if (text.includes("export class")) {
    return "class_definition";
  } else if (
    text.includes("export function") ||
    text.includes("export async function")
  ) {
    return "function_definition";
  } else if (/\w+\s*\([^)]*\)\s*{/.test(text)) {
    return "method_definition";
  }
  return "file";
}

/**
 * Extract function or class name from the chunk
 * @param text - The chunk text content
 * @param source - The source file name
 * @returns The extracted function/class name or file name
 */
function getFunctionName(text: string, source: string): string {
  // Extract class name
  const classMatch = text.match(/export\s+class\s+(\w+)/);
  if (classMatch) {
    return classMatch[1];
  }

  // Extract function name
  const functionMatch = text.match(/export\s+(?:async\s+)?function\s+(\w+)/);
  if (functionMatch) {
    return functionMatch[1];
  }

  // Extract method name (with class context if available)
  const methodMatch = text.match(/(?:async\s+)?(\w+)\s*\([^)]*\)\s*{/);
  if (methodMatch && methodMatch[1] !== "constructor") {
    // Try to find class context
    const classContext = text.match(/class\s+(\w+)/);
    if (classContext) {
      return `${classContext[1]}.${methodMatch[1]}`;
    }
    return methodMatch[1];
  }

  // Default to source file name
  return source;
}

// Run the upsert function
upsertCodeDocuments().catch(console.error);
