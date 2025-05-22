import { mastra } from "../mastra";

// Example: Agent choosing basicSearchTool for simple keyword searches
async function basicSearchExample() {
  const basicAgent = mastra.getAgent("basicAgent");
  // Example 1: Authentication implementation
  const authResponse = await basicAgent.generate(
    "How do we implement JWT authentication in our system?"
  );
  console.log("\nJWT Authentication Search:", authResponse.text);

  // Example 2: Error handling patterns
  const errorResponse = await basicAgent.generate(
    "What are our standard error handling patterns for API responses?"
  );
  console.log("\nError Handling Search:", errorResponse.text);

  // Example 3: Configuration structure
  const configResponse = await basicAgent.generate(
    "What monitoring configurations are available in our system?"
  );
  console.log("\nConfiguration Search:", configResponse.text);

  process.exit(0);
}

basicSearchExample().catch(console.error);

/* Example output:
JWT Authentication Search: Based on our documentation, JWT authentication is implemented using the 
AuthenticationService class. It handles token generation, verification, and session validation...

Error Handling Search: Our system uses a standardized error handling approach with custom error 
classes like AppError and HttpError. All errors include a code, message, and optional details...

Configuration Search: The monitoring configuration includes APM settings, logging outputs 
(Elasticsearch and CloudWatch), metrics collection, and alert configurations for Slack and PagerDuty...
*/
