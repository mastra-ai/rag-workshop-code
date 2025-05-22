import { mastra } from "../mastra";

// Example: Agent choosing findCodeTool for code-specific searches
async function findCodeExample() {
  const codeAgent = mastra.getAgent("codeAgent");
  // Example 1: Finding authentication method implementation
  const authResponse = await codeAgent.generate(
    "Find the implementation of the verifyToken method in our authentication system"
  );
  console.log("\nAuth Implementation Search:", authResponse.text);

  // Example 2: Finding logger class definition
  const loggerResponse = await codeAgent.generate(
    "Show me the AuditLogger class definition"
  );
  console.log("\nLogger Class Search:", loggerResponse.text);

  // Example 3: Finding error handling function
  const errorResponse = await codeAgent.generate(
    "Find the withDbErrorHandling function implementation in our error handling system"
  );
  console.log("\nError Handling Function Search:", errorResponse.text);

  process.exit(0);
}

findCodeExample().catch(console.error);

// Example output:
//
// Auth Implementation Search:
// "I found the verifyToken method implementation in the AuthenticationService class that verifies JWT tokens
// and validates sessions. It handles different error cases like expired tokens and invalid tokens with
// specific error messages."
//
// Logger Class Search:
// "I found the AuditLogger class that extends the base Logger class and provides specialized methods for
// tracking data changes. The logDataChange method records entity modifications with before/after states,
// which is useful for audit trails and compliance requirements."
//
// Error Handling Function Search:
// "I found the withDbErrorHandling function that wraps database operations with error handling. It takes
// an async operation function and an error message, then executes the operation in a try/catch block.
// If an error occurs, it wraps it in a DatabaseError with additional context, making it easier to track
// and debug database issues."
