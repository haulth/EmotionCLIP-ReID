const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

try {
  const [graphPath, layersPath, outputPath] = process.argv.slice(2);
  if (!graphPath || !layersPath || !outputPath) {
    fail('Usage: node ua-tour-prepare.js <assembled-graph.json> <layers.json> <output.json>');
  }

  const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
  const layersDocument = JSON.parse(fs.readFileSync(layersPath, 'utf8'));
  const layers = Array.isArray(layersDocument) ? layersDocument : (layersDocument.layers || []);
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges) || !Array.isArray(layers)) {
    fail('Invalid graph or layers document.');
  }

  const input = {
    nodes: graph.nodes,
    edges: graph.edges,
    layers: layers.map(({ id, name, description }) => ({ id, name, description })),
  };
  fs.writeFileSync(outputPath, `${JSON.stringify(input, null, 2)}\n`, 'utf8');
} catch (error) {
  fail(error && error.stack ? error.stack : String(error));
}
