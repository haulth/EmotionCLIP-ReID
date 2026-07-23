const fs = require('fs');

const graphPath = process.argv[2];
const outputPath = process.argv[3];
if (!graphPath || !outputPath) {
  console.error('Usage: node ua-arch-prepare.js <assembled-graph.json> <ua-arch-input.json>');
  process.exit(1);
}

try {
  const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
  const fileLevelTypes = new Set([
    'file', 'config', 'document', 'service', 'pipeline',
    'table', 'schema', 'resource', 'endpoint',
  ]);
  const fileNodes = graph.nodes.filter((node) => fileLevelTypes.has(node.type));
  const fileIds = new Set(fileNodes.map((node) => node.id));
  const allEdges = graph.edges.filter(
    (edge) => fileIds.has(edge.source) && fileIds.has(edge.target),
  );
  const importEdges = allEdges.filter((edge) => edge.type === 'imports');
  fs.writeFileSync(
    outputPath,
    JSON.stringify({ fileNodes, importEdges, allEdges }, null, 2) + '\n',
    'utf8',
  );
  console.log(`Prepared ${fileNodes.length} file nodes, ${importEdges.length} import edges, ${allEdges.length} total file-level edges.`);
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
