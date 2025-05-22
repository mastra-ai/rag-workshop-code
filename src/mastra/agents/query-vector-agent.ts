import { Agent } from "@mastra/core/agent";
import { queryVectorTool } from "../tools";
import { openai } from "@ai-sdk/openai";
import { PGVECTOR_PROMPT } from "@mastra/pg";

export const queryVectorAgent = new Agent({
  name: "queryVectorAgent",
  instructions: `
  Helps explore and understand codebases using RAG with vector search.
  
  Filter the context by searching the metadata.
  
  The metadata is structured as follows:
  
  {
    source: string, // filename (e.g., 'auth.md', 'application-settings.json')
    type: string,   // 'documentation' or 'configuration'
    section: string, // 'authentication', 'error-handling', 'logging', or 'settings'
    format: string, // 'markdown' or 'json'
    text: string    // the actual content of the chunk
  }
  
  ${PGVECTOR_PROMPT}
  `,
  model: openai("gpt-4o"),
  tools: {
    queryVectorTool,
  },
});
