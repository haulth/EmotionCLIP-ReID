const fs = require('fs');

try {
  const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
  const nodes = input.nodes || [];
  const edges = input.edges || [];
  const layers = input.layers || [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));
  const adjacency = new Map(nodes.map((node) => [node.id, new Set()]));

  for (const edge of edges) {
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
    if (adjacency.has(edge.source) && byId.has(edge.target)) adjacency.get(edge.source).add(edge.target);
  }

  const rank = (map, field) => [...map]
    .map(([id, value]) => ({ id, [field]: value, name: byId.get(id)?.name || id }))
    .sort((left, right) => right[field] - left[field] || left.id.localeCompare(right.id))
    .slice(0, 20);
  const outValues = [...fanOut.values()].sort((a, b) => a - b);
  const inValues = [...fanIn.values()].sort((a, b) => a - b);
  const out90 = outValues[Math.floor(outValues.length * 0.9)] || 0;
  const in25 = inValues[Math.floor(inValues.length * 0.25)] || 0;

  const entryPointCandidates = [];
  const codeNames = /^(index\.(ts|js)|main\.(ts|js|go|py|rs|cpp|c)|app\.(ts|js|py)|server\.(ts|js)|mod\.rs|manage\.py|wsgi\.py|asgi\.py|run\.py|__main__\.py|Application\.java|Main\.java|Program\.cs|config\.ru|index\.php|App\.swift|Application\.kt)$/;
  for (const node of nodes) {
    let score = 0;
    const filePath = node.filePath || '';
    if (node.type === 'document' && filePath === 'README.md') score += 5;
    else if (node.type === 'document' && /^[^/]+\.md$/i.test(filePath)) score += 2;
    if (node.type === 'file') {
      if (codeNames.test(node.name || '')) score += 3;
      if ((filePath.match(/\//g) || []).length <= 1) score += 1;
      if ((fanOut.get(node.id) || 0) >= out90) score += 1;
      if ((fanIn.get(node.id) || 0) <= in25) score += 1;
    }
    if (score) entryPointCandidates.push({ id: node.id, score, name: node.name, summary: node.summary });
  }
  entryPointCandidates.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));

  const preferred = byId.has('file:train_emotionclip.py')
    ? 'file:train_emotionclip.py'
    : entryPointCandidates.find((item) => byId.get(item.id)?.type === 'file')?.id;
  const depthMap = {};
  const order = [];
  const byDepth = {};
  if (preferred) {
    const queue = [preferred];
    depthMap[preferred] = 0;
    while (queue.length) {
      const current = queue.shift();
      const depth = depthMap[current];
      order.push(current);
      (byDepth[depth] ||= []).push(current);
      for (const edge of edges) {
        if (edge.source === current && ['imports', 'calls'].includes(edge.type)
          && byId.has(edge.target) && depthMap[edge.target] === undefined) {
          depthMap[edge.target] = depth + 1;
          queue.push(edge.target);
        }
      }
    }
  }

  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const node of nodes) {
    const item = { id: node.id, name: node.name, type: node.type, summary: node.summary };
    if (node.type === 'document') nonCodeFiles.documentation.push(item);
    else if (['service', 'pipeline', 'resource'].includes(node.type)) nonCodeFiles.infrastructure.push(item);
    else if (['table', 'schema', 'endpoint'].includes(node.type)) nonCodeFiles.data.push(item);
    else if (node.type === 'config') nonCodeFiles.config.push(item);
  }

  const coupledTypes = new Set(['imports', 'calls']);
  const directed = new Set(edges.filter((edge) => coupledTypes.has(edge.type))
    .map((edge) => `${edge.source}\0${edge.target}`));
  const clusterMap = new Map();
  for (const edge of edges) {
    if (!coupledTypes.has(edge.type) || edge.source >= edge.target
      || !directed.has(`${edge.target}\0${edge.source}`)) continue;
    const members = new Set([edge.source, edge.target]);
    let changed = true;
    while (changed && members.size < 5) {
      changed = false;
      const candidates = nodes
        .filter((node) => !members.has(node.id))
        .map((node) => ({
          id: node.id,
          links: [...members].filter((member) =>
            adjacency.get(node.id)?.has(member) || adjacency.get(member)?.has(node.id)).length,
        }))
        .filter((candidate) => candidate.links >= 2)
        .sort((a, b) => b.links - a.links || a.id.localeCompare(b.id));
      if (candidates.length) {
        members.add(candidates[0].id);
        changed = true;
      }
    }
    const memberList = [...members].sort();
    const key = memberList.join('\0');
    const edgeCount = edges.filter((candidate) =>
      members.has(candidate.source) && members.has(candidate.target)).length;
    const previous = clusterMap.get(key);
    if (!previous || previous.edgeCount < edgeCount) clusterMap.set(key, { nodes: memberList, edgeCount });
  }
  const clusters = [...clusterMap.values()]
    .sort((a, b) => b.edgeCount - a.edgeCount || a.nodes.join().localeCompare(b.nodes.join()))
    .slice(0, 10);

  const nodeSummaryIndex = {};
  for (const node of nodes) {
    nodeSummaryIndex[node.id] = { name: node.name, type: node.type, summary: node.summary };
  }
  const result = {
    scriptCompleted: true,
    entryPointCandidates: entryPointCandidates.slice(0, 5),
    fanInRanking: rank(fanIn, 'fanIn'),
    fanOutRanking: rank(fanOut, 'fanOut'),
    bfsTraversal: { startNode: preferred, order, depthMap, byDepth },
    nonCodeFiles,
    clusters,
    layers: { count: layers.length, list: layers.map(({ id, name, description }) => ({ id, name, description })) },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };
  fs.writeFileSync(process.argv[3], JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error.stack || String(error));
  process.exit(1);
}
