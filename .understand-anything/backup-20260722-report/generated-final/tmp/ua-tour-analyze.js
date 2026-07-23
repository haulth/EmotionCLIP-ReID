const fs = require('fs');

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function sortedRanking(nodes, valueMap, field) {
  return nodes
    .map((node) => ({ id: node.id, [field]: valueMap.get(node.id)?.size || 0, name: node.name }))
    .sort((a, b) => b[field] - a[field] || a.id.localeCompare(b.id))
    .slice(0, 20);
}

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) {
    fail('Usage: node ua-tour-analyze.js <input.json> <results.json>');
  }

  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = Array.isArray(input.nodes) ? input.nodes : [];
  const edges = Array.isArray(input.edges) ? input.edges : [];
  const layers = Array.isArray(input.layers) ? input.layers : [];
  if (!nodes.length) fail('Input contains no nodes.');

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const incoming = new Map(nodes.map((node) => [node.id, new Set()]));
  const outgoing = new Map(nodes.map((node) => [node.id, new Set()]));
  for (const edge of edges) {
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target) || edge.source === edge.target) continue;
    outgoing.get(edge.source).add(edge.target);
    incoming.get(edge.target).add(edge.source);
  }

  const fanInRanking = sortedRanking(nodes, incoming, 'fanIn');
  const fanOutRanking = sortedRanking(nodes, outgoing, 'fanOut');
  const codeNodes = nodes.filter((node) => node.type === 'file');
  const fanOutValues = codeNodes.map((node) => outgoing.get(node.id)?.size || 0).sort((a, b) => a - b);
  const fanInValues = codeNodes.map((node) => incoming.get(node.id)?.size || 0).sort((a, b) => a - b);
  const percentileAt = (values, percentile) => values.length
    ? values[Math.max(0, Math.min(values.length - 1, Math.ceil(percentile * values.length) - 1))]
    : 0;
  const highFanOutCutoff = percentileAt(fanOutValues, 0.9);
  const lowFanInCutoff = percentileAt(fanInValues, 0.25);
  const entryNames = new Set([
    'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js', 'server.ts', 'server.js',
    'mod.rs', 'main.go', 'main.py', 'main.rs', 'manage.py', 'app.py', 'wsgi.py', 'asgi.py',
    'run.py', '__main__.py', 'application.java', 'main.java', 'program.cs', 'config.ru',
    'index.php', 'app.swift', 'application.kt', 'main.cpp', 'main.c',
  ]);
  const entryPointCandidates = [];
  for (const node of nodes) {
    const path = String(node.filePath || node.name || '').replace(/\\/g, '/');
    const name = String(node.name || path.split('/').pop() || '');
    let score = 0;
    if (node.type === 'file') {
      if (entryNames.has(name.toLowerCase())) score += 3;
      if (path.split('/').filter(Boolean).length <= 2) score += 1;
      if ((outgoing.get(node.id)?.size || 0) >= highFanOutCutoff) score += 1;
      if ((incoming.get(node.id)?.size || 0) <= lowFanInCutoff) score += 1;
    } else if (node.type === 'document') {
      if (path.toLowerCase() === 'readme.md') score += 5;
      else if (!path.includes('/') && name.toLowerCase().endsWith('.md')) score += 2;
    }
    if (score > 0) {
      entryPointCandidates.push({ id: node.id, score, name, summary: node.summary || '' });
    }
  }
  entryPointCandidates.sort((a, b) =>
    b.score - a.score
    || (outgoing.get(b.id)?.size || 0) - (outgoing.get(a.id)?.size || 0)
    || a.id.localeCompare(b.id));
  entryPointCandidates.splice(5);

  const codeStart = entryPointCandidates.find((candidate) => nodeById.get(candidate.id)?.type === 'file');
  const bfsTraversal = { startNode: codeStart?.id || null, order: [], depthMap: {}, byDepth: {} };
  if (codeStart) {
    const traversable = new Map();
    for (const edge of edges) {
      if (!['imports', 'calls'].includes(edge.type)) continue;
      if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
      if (!traversable.has(edge.source)) traversable.set(edge.source, new Set());
      traversable.get(edge.source).add(edge.target);
    }
    const queue = [codeStart.id];
    bfsTraversal.depthMap[codeStart.id] = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      bfsTraversal.order.push(current);
      const depth = bfsTraversal.depthMap[current];
      if (!bfsTraversal.byDepth[String(depth)]) bfsTraversal.byDepth[String(depth)] = [];
      bfsTraversal.byDepth[String(depth)].push(current);
      const neighbors = [...(traversable.get(current) || [])].sort();
      for (const neighbor of neighbors) {
        if (Object.prototype.hasOwnProperty.call(bfsTraversal.depthMap, neighbor)) continue;
        bfsTraversal.depthMap[neighbor] = depth + 1;
        queue.push(neighbor);
      }
    }
  }

  const projectNode = (node) => ({ id: node.id, name: node.name, type: node.type, summary: node.summary || '' });
  const nonCodeFiles = {
    documentation: nodes.filter((node) => node.type === 'document').map(projectNode),
    infrastructure: nodes.filter((node) => ['service', 'pipeline', 'resource'].includes(node.type)).map(projectNode),
    data: nodes.filter((node) => ['table', 'schema', 'endpoint'].includes(node.type)).map(projectNode),
    config: nodes.filter((node) => node.type === 'config').map(projectNode),
  };

  const relationKeys = new Set();
  const connectiveEdges = [];
  for (const edge of edges) {
    if (!['imports', 'calls'].includes(edge.type)) continue;
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    relationKeys.add(`${edge.type}\u0000${edge.source}\u0000${edge.target}`);
    connectiveEdges.push(edge);
  }
  const seeds = [];
  for (const edge of connectiveEdges) {
    if (edge.source.localeCompare(edge.target) >= 0) continue;
    if (relationKeys.has(`${edge.type}\u0000${edge.target}\u0000${edge.source}`)) {
      seeds.push([edge.source, edge.target]);
    }
  }
  const undirected = new Map(nodes.map((node) => [node.id, new Set()]));
  for (const edge of connectiveEdges) {
    undirected.get(edge.source).add(edge.target);
    undirected.get(edge.target).add(edge.source);
  }
  const clusterMap = new Map();
  for (const seed of seeds) {
    const cluster = new Set(seed);
    let expanded = true;
    while (expanded && cluster.size < 5) {
      expanded = false;
      const candidates = nodes
        .filter((node) => !cluster.has(node.id))
        .map((node) => ({
          id: node.id,
          links: [...cluster].filter((member) => undirected.get(node.id)?.has(member)).length,
        }))
        .filter((candidate) => candidate.links >= 2)
        .sort((a, b) => b.links - a.links || a.id.localeCompare(b.id));
      if (candidates.length) {
        cluster.add(candidates[0].id);
        expanded = true;
      }
    }
    const ids = [...cluster].sort();
    const key = ids.join('\u0000');
    if (clusterMap.has(key)) continue;
    const idSet = new Set(ids);
    const edgeCount = connectiveEdges.filter((edge) => idSet.has(edge.source) && idSet.has(edge.target)).length;
    clusterMap.set(key, { nodes: ids, edgeCount });
  }
  const clusters = [...clusterMap.values()]
    .sort((a, b) => b.edgeCount - a.edgeCount || b.nodes.length - a.nodes.length || a.nodes[0].localeCompare(b.nodes[0]))
    .slice(0, 10);

  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [
    node.id,
    { name: node.name, type: node.type, summary: node.summary || '' },
  ]));

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal,
    nonCodeFiles,
    clusters,
    layers: {
      count: layers.length,
      list: layers.map(({ id, name, description }) => ({ id, name, description })),
    },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
} catch (error) {
  fail(error && error.stack ? error.stack : String(error));
}
