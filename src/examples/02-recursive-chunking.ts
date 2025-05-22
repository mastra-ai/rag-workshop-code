import { MDocument } from "@mastra/rag";

const codeContent = `
function processData(input: string) {
  // Process the input
  const result = transform(input);
  return result;
}

function transform(data: string) {
  return data.toUpperCase();
}
`;

const codeDoc = new MDocument({
  docs: [{ text: codeContent }],
  type: "text",
});

const chunks = await codeDoc.chunk({
  strategy: "recursive",
  size: 150,
  overlap: 0,
});

console.log("Code Chunks:", chunks);