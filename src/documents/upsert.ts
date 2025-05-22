import { MDocument } from "@mastra/rag";
import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import { mastra } from "../mastra";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from 'url';

// Get the directory name of the current file
// This works in both CommonJS and ES modules
const currentFilePath = fileURLToPath(import.meta.url);
const currentDir = path.dirname(currentFilePath);

// Go up to the project root (1 level up from src/documents)
const projectRoot = path.resolve(currentDir, '../..');


async function upsertDocuments() {
  // Use the absolute path to the documents directory
  const docsDir = path.join(projectRoot, "src", "documents");

  // Process markdown documents
  const markdownDocs = [
    {
      path: path.join(docsDir, "auth.md"),
      type: "documentation",
      section: "authentication",
    },
    {
      path: path.join(docsDir, "error-handling.md"),
      type: "documentation",
      section: "error-handling",
    },
    {
      path: path.join(docsDir, "logging.md"),
      type: "documentation",
      section: "logging",
    },
  ];

  // Read and process markdown files
  const markdownContents = await Promise.all(
    markdownDocs.map(async (doc) => ({
      text: await fs.readFile(doc.path, "utf-8"),
      metadata: {
        source: path.basename(doc.path),
        type: doc.type,
        section: doc.section,
        format: "markdown",
      },
    }))
  );

  // Process markdown documents
  const markdownDoc = new MDocument({
    docs: markdownContents,
    type: "markdown",
  });

  await markdownDoc.chunk({
    strategy: "markdown",
    headers: [
      ["#", "Header 1"],
      ["##", "Header 2"],
    ],
    maxSize: 1500, // Large enough for complete code examples
    minSize: 500, // Small enough to be specific but meaningful
    overlap: 100, // Enough to maintain context between chunks
  });

  // Process JSON configuration
  const configPath = path.join(docsDir, "application-settings.json");
  const configContent = await fs.readFile(configPath, "utf-8");
  const configDoc = new MDocument({
    docs: [
      {
        text: configContent,
        metadata: {
          source: "application-settings.json",
          type: "configuration",
          section: "settings",
          format: "json",
        },
      },
    ],
    type: "json",
  });

  await configDoc.chunk({
    strategy: "json",
    maxSize: 1000, // Smaller for JSON as each chunk should be a logical unit
    minSize: 200, // Allow smaller chunks for individual config sections
    overlap: 50, // Less overlap needed for structured data
  });

  // Combine chunks from both document types
  const markdownChunks = markdownDoc.getDocs();
  const jsonChunks = configDoc.getDocs();
  const allChunks = [...markdownChunks, ...jsonChunks];

  // Generate embeddings for all chunks
  const { embeddings } = await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: allChunks.map((chunk) => chunk.text),
  });

  // Get PgVector instance
  const pgVector = mastra.getVector("pg");
  // Delete existing index (if exists)
  await pgVector.deleteIndex({ indexName: "workshop" });
  await pgVector.createIndex({
    indexName: "workshop",
    dimension: 1536,
  });

  // Upsert vectors
  await pgVector.upsert({
    indexName: "workshop",
    vectors: embeddings,
    metadata: allChunks.map((chunk) => ({
      ...chunk.metadata,
      text: chunk.text,
    })),
  });

  console.log(`Successfully upserted ${allChunks.length} document chunks`);

  // Log some stats about what was processed
  const stats = allChunks.reduce(
    (acc, chunk) => {
      const format = chunk.metadata.format;
      acc[format] = (acc[format] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  console.log("\nChunks by format:");
  Object.entries(stats).forEach(([format, count]) => {
    console.log(`${format}: ${count} chunks`);
  });
}

// Run the upsert
upsertDocuments().catch(console.error);
