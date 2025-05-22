# RAG Workshop Examples

This repository contains examples demonstrating RAG (Retrieval Augmented Generation) capabilities using Mastra.

## Getting Started

1. **Install dependencies**:
```bash
pnpm i
```

2. **Set up environment variables**:
   - Copy the example environment file to create your own:
```bash
cp .env.example .env
```
   - Open the `.env` file and add your:
     - OpenAI API key
     - PostgreSQL connection string
     - Cohere API key (if using Cohere models)

## Structure

### Sample Documents (`src/documents/`)

Contains markdown and json documents used to test our RAG functionality:

- `auth.md` - Authentication guide with JWT examples
- `error-handling.md` - Error handling patterns and best practices
- `logging.md` - Logging system documentation
- `application-settings.json` - Application settings configuration
- `upsert.ts` - Script to insert documents into the vector store

### Basic Chunking Examples (`src/examples/01-04`)

1. **Character Chunking** (`01-character-chunking.ts`)
   - Simple text chunking by character count
   - Shows basic overlapping chunks

2. **Recursive Chunking** (`02-recursive-chunking.ts`)
   - Code-aware chunking that preserves function boundaries
   - Demonstrates chunking while maintaining code context

3. **JSON Chunking** (`03-json-chunking.ts`)
   - Chunks JSON while preserving valid structure
   - Shows how to handle structured data formats

4. **Markdown Chunking** (`04-markdown-chunking.ts`)
   - Header-aware markdown chunking
   - Demonstrates hierarchical document splitting

5. **Embeddings** (`05-embedding.ts`)
   - Different embedding models and their characteristics
   - Single vs batch embedding
   - Comparing embedding dimensions and quality

### Vector Store Operations (`src/examples/06-09`)

6. **Vector Upserting** (`06-vector-upserting.ts`)
   - How to insert documents into PgVector
   - Complete flow: chunking → embedding → storing

7. **Vector Update** (`07-vector-update-delete.ts`)
   - How to update and delete documents in PgVector
   - Complete flow: updating → embedding → updating

8. **Vector Search** (`08-vector-search.ts`)
   - Basic vector similarity search
   - Shows simple querying with filters

9. **Vector Reranking** (`09-vector-reranking.ts`)
   - Advanced search with result reranking
   - Demonstrates improving result relevance

### Agent Examples (`src/examples/10-11`)

10. **Basic Search** (`10-basic-search-usage.ts`)
   - Simple keyword search through an agent
   - Example: "What does our documentation say about authentication?"

11. **Vector Search** (`11-query-vector-usage.ts`)
   - Semantic search through an agent with:
     - Basic search
     - Filtered search (by file type)
     - Reranked search for better results

### Bonus Examples (`src/bonus/`)

- **Advanced Code Usage** (`01-find-code-usage.ts`)
  - Code-aware search through an agent
  - Examples:
    - Finding function definitions
    - Finding usage examples

## Usage

1. Try the chunking examples to understand different strategies:
```bash
pnpm tsx src/examples/01-character-chunking.ts
pnpm tsx src/examples/02-recursive-chunking.ts
pnpm tsx src/examples/03-json-chunking.ts
pnpm tsx src/examples/04-markdown-chunking.ts
```

2. Try the embedding and vector store operations:
```bash
pnpm tsx src/examples/05-embedding.ts
pnpm tsx src/examples/06-vector-upserting.ts
pnpm tsx src/examples/07-vector-update-delete.ts
pnpm tsx src/examples/08-vector-search.ts
pnpm tsx src/examples/09-vector-reranking.ts
```

3. Explore the agent examples to see RAG in action:

Upsert the example data for the agent examples:
```bash
pnpm tsx src/documents/upsert.ts
```

```bash
pnpm tsx src/examples/10-basic-search-usage.ts
pnpm tsx src/examples/11-query-vector-usage.ts
```

## Key Concepts

### Chunking
- Breaking documents into meaningful pieces
- Different strategies for different content types
- Preserving context and structure

### Vector Operations
- Converting text to embeddings
- Storing in PgVector
- Similarity search
- Result reranking

### Agent Tools
- Basic keyword search
- Semantic search with filters

## Bonus Example (`src/bonus/`)

A bonus example that demonstrates how to use RAG to search through code files, find implementations of specific methods, and locate class definitions in your codebase.

### Bonus Documents (`src/bonus/documents/`)

Contains markdown documents used to test our RAG functionality:

- `authentication-service.ts` - Authentication service implementation
- `error-handling.ts` - Error handling patterns and best practices
- `logger.ts` - Logging system documentation
- `upsert.ts` - Script to insert documents into the vector store

Upsert the code documents for the bonus examples:
```bash
pnpm tsx src/bonus/documents/upsert.ts
```

Run the code search example:
```bash
pnpm tsx src/bonus/01-find-code-usage.ts
```
