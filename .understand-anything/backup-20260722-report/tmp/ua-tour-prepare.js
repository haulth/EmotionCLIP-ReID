const fs = require('fs');

try {
  const graph = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
  const layerData = JSON.parse(fs.readFileSync(process.argv[3], 'utf8').replace(/^\uFEFF/, ''));
  const layers = Array.isArray(layerData) ? layerData : (layerData.layers || []);
  const input = {
    nodes: graph.nodes || [],
    edges: graph.edges || [],
    layers,
  };
  fs.writeFileSync(process.argv[4], JSON.stringify(input, null, 2));
} catch (error) {
  console.error(error.stack || String(error));
  process.exit(1);
}
