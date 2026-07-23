const fs = require('fs');

const [graphPath, outputPath] = process.argv.slice(2);
if (!graphPath || !outputPath) {
  console.error('Usage: node ua-arch-prepare.js <graph.json> <output.json>');
  process.exit(1);
}

const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8').replace(/^\uFEFF/, ''));
const fileLevelTypes = new Set([
  'file', 'config', 'document', 'service', 'pipeline', 'table', 'schema',
  'resource', 'endpoint',
]);
const fileNodes = graph.nodes.filter(
  (node) => fileLevelTypes.has(node.type) && typeof node.filePath === 'string',
);
const fileIds = new Set(fileNodes.map((node) => node.id));
const allEdges = graph.edges.filter(
  (edge) => fileIds.has(edge.source) && fileIds.has(edge.target),
);
const importEdges = allEdges.filter((edge) => edge.type === 'imports');

fs.writeFileSync(outputPath, JSON.stringify({ fileNodes, importEdges, allEdges }, null, 2));
