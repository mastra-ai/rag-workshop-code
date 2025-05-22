import { mastra } from "../mastra";

// Example: Agent choosing queryVectorTool for semantic searches
async function queryVectorExample() {
  const queryVectorAgent = mastra.getAgent("queryVectorAgent");
  // Example 1: Semantic search for error handling
  const errorResponse = await queryVectorAgent.generate(
    "Find implementations in our codebase that show how we handle database errors and validation errors"
  );

  console.log("\nError Handling Search:", errorResponse.text);

  // Example 2: Search with format and section filters
  const configResponse = await queryVectorAgent.generate(
    "How do we handle errors in our application? Filter where section is error-handling AND format is markdown. Return the top 3 results."
  );

  console.log("\nFiltered Error Handling Search:", configResponse.text);

  // Example 3: Search with reranking for authentication
  const authResponse = await queryVectorAgent.generate(
    "Find the most relevant code examples showing JWT token verification and session validation."
  );

  console.log("\nAuthentication Search:", authResponse.text);

  process.exit(0);
}

queryVectorExample().catch(console.error);

/* Example output:

Error Handling Search: I found several implementations for handling database and validation errors. 
The DatabaseError class extends AppError and captures the original error with context. It's used 
throughout our codebase to standardize database error handling...

Filtered Error Handling Search: According to our error-handling documentation, we use a centralized 
approach with the errorHandlerMiddleware. This middleware categorizes errors by type, logs them 
appropriately, and returns standardized error responses...

Authentication Search: The JWT token verification and session validation are implemented in the 
AuthenticationService class. The verifyToken method uses jwt.verify to decode and validate tokens, 
while validateSession checks if the session exists and hasn't expired...
*/
