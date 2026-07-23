const fs = require('fs');

const [graphPath, changedPath, outputPath] = process.argv.slice(2);
if (!graphPath || !changedPath || !outputPath) {
  throw new Error('usage: node ua-prune-existing.cjs <graph> <changed-files> <output>');
}

const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8').replace(/^\uFEFF/, ''));
const changed = new Set(
  fs.readFileSync(changedPath, 'utf8')
    .split(/\r?\n/)
    .map((value) => value.trim().replace(/\\/g, '/'))
    .filter(Boolean),
);
const removedIds = new Set(
  graph.nodes
    .filter((node) => node.filePath && changed.has(String(node.filePath).replace(/\\/g, '/')))
    .map((node) => node.id),
);
const nodes = graph.nodes.filter((node) => !removedIds.has(node.id));
const edges = graph.edges.filter(
  (edge) => !removedIds.has(edge.source) && !removedIds.has(edge.target),
);
fs.writeFileSync(outputPath, JSON.stringify({ nodes, edges }, null, 2));
process.stdout.write(
  `Pruned ${removedIds.size} nodes and ${graph.edges.length - edges.length} edges; retained ${nodes.length} nodes and ${edges.length} edges.\n`,
);
