const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

try {
  const [resultsPath, tourPath] = process.argv.slice(2);
  if (!resultsPath || !tourPath) fail('Usage: node ua-tour-validate.js <results.json> <tour.json>');
  const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
  const tour = JSON.parse(fs.readFileSync(tourPath, 'utf8'));
  const validNodeIds = new Set(Object.keys(results.nodeSummaryIndex || {}));

  if (!Array.isArray(tour) || tour.length < 5 || tour.length > 15) {
    fail('Tour must contain 5-15 steps.');
  }
  let nonCodeStops = 0;
  for (let index = 0; index < tour.length; index += 1) {
    const step = tour[index];
    if (step.order !== index + 1) fail(`Invalid order at index ${index}.`);
    if (typeof step.title !== 'string' || !step.title.trim()) fail(`Missing title at step ${step.order}.`);
    if (typeof step.description !== 'string' || !step.description.trim()) fail(`Missing description at step ${step.order}.`);
    if (!Array.isArray(step.nodeIds) || step.nodeIds.length < 1 || step.nodeIds.length > 5) {
      fail(`Step ${step.order} must reference 1-5 nodes.`);
    }
    for (const nodeId of step.nodeIds) {
      if (!validNodeIds.has(nodeId)) fail(`Unknown node ID at step ${step.order}: ${nodeId}`);
      if (results.nodeSummaryIndex[nodeId].type !== 'file') nonCodeStops += 1;
    }
  }
  if (tour[0].nodeIds[0] !== 'document:README.md') fail('Step 1 must start with README.md.');
  if (nonCodeStops < 2) fail('Tour must include at least two non-code references.');
  process.stdout.write(JSON.stringify({ valid: true, steps: tour.length, nonCodeReferences: nonCodeStops }));
} catch (error) {
  fail(error && error.stack ? error.stack : String(error));
}
