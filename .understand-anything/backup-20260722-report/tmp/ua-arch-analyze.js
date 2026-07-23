const fs = require('fs');
const path = require('path');

function fail(message) {
  console.error(message);
  process.exit(1);
}

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) fail('Usage: node ua-arch-analyze.js <input.json> <output.json>');

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8').replace(/^\uFEFF/, ''));
  const fileNodes = input.fileNodes || [];
  const importEdges = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const nodeById = new Map(fileNodes.map((node) => [node.id, node]));

  const normalizedParts = fileNodes.map((node) => node.filePath.replace(/\\/g, '/').split('/'));
  let commonPrefixParts = normalizedParts.length ? [...normalizedParts[0].slice(0, -1)] : [];
  for (const parts of normalizedParts.slice(1)) {
    const dirs = parts.slice(0, -1);
    let index = 0;
    while (index < commonPrefixParts.length && index < dirs.length && commonPrefixParts[index] === dirs[index]) index += 1;
    commonPrefixParts = commonPrefixParts.slice(0, index);
  }

  const isFlat = normalizedParts.every((parts) => parts.length === commonPrefixParts.length + 1);
  const filePattern = (filePath) => {
    const normalized = filePath.replace(/\\/g, '/');
    const base = path.posix.basename(normalized);
    const lower = normalized.toLowerCase();
    if (/(^|\/)(test_[^/]+\.py|[^/]+\.(test|spec)\.[^/]+|[^/]+_test\.go|[^/]+test\.java|[^/]+_spec\.rb|[^/]+tests\.cs)$/.test(lower)) return 'test';
    if (lower.endsWith('.d.ts')) return 'types';
    if (['index.ts', 'index.js', '__init__.py'].includes(base) && normalized.split('/').length > 1) return 'entry';
    if (['manage.py', 'config.ru'].includes(base)) return 'entry';
    if (['wsgi.py', 'asgi.py'].includes(base)) return 'config';
    if (['cargo.toml', 'go.mod', 'gemfile', 'pom.xml', 'build.gradle', 'composer.json'].includes(lower)) return 'config';
    if (base === 'Dockerfile' || lower.includes('docker-compose')) return 'infrastructure';
    if (/\.tf(vars)?$/.test(lower)) return 'infrastructure';
    if (lower.startsWith('.github/workflows/') || lower === '.gitlab-ci.yml' || base === 'Jenkinsfile') return 'ci-cd';
    if (lower.endsWith('.sql')) return 'data';
    if (/\.(graphql|gql|proto)$/.test(lower)) return 'types';
    if (/\.(md|rst)$/.test(lower)) return 'documentation';
    if (base === 'Makefile') return 'infrastructure';
    return null;
  };

  const directoryGroups = {};
  const groupById = {};
  for (let index = 0; index < fileNodes.length; index += 1) {
    const node = fileNodes[index];
    const parts = normalizedParts[index];
    const relative = parts.slice(commonPrefixParts.length);
    let group;
    if (isFlat) group = filePattern(node.filePath) || path.posix.extname(node.filePath).slice(1) || 'root';
    else group = relative.length > 1 ? relative[0] : 'root';
    directoryGroups[group] ||= [];
    directoryGroups[group].push(node.id);
    groupById[node.id] = group;
  }

  const nodeTypeGroups = {};
  for (const node of fileNodes) {
    nodeTypeGroups[node.type] ||= [];
    nodeTypeGroups[node.type].push(node.id);
  }

  const fileFanIn = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const groupImportsFrom = {};
  const groupImportedBy = {};
  const interCounts = new Map();
  for (const group of Object.keys(directoryGroups)) {
    groupImportsFrom[group] = new Set();
    groupImportedBy[group] = new Set();
  }
  for (const edge of importEdges) {
    fileFanOut[edge.source] = (fileFanOut[edge.source] || 0) + 1;
    fileFanIn[edge.target] = (fileFanIn[edge.target] || 0) + 1;
    const from = groupById[edge.source];
    const to = groupById[edge.target];
    if (!from || !to) continue;
    groupImportsFrom[from].add(to);
    groupImportedBy[to].add(from);
    const key = `${from}\u0000${to}`;
    interCounts.set(key, (interCounts.get(key) || 0) + 1);
  }
  const importAdjacency = {};
  for (const node of fileNodes) importAdjacency[node.id] = [];
  for (const edge of importEdges) importAdjacency[edge.source].push(edge.target);

  const crossCounts = new Map();
  const nonCodeConnections = [];
  for (const edge of allEdges) {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (!source || !target) continue;
    if (source.type !== target.type) {
      const key = `${source.type}\u0000${target.type}\u0000${edge.type}`;
      crossCounts.set(key, (crossCounts.get(key) || 0) + 1);
    }
    if (source.type !== 'file' || target.type !== 'file') {
      nonCodeConnections.push({ source: edge.source, target: edge.target, edgeType: edge.type });
    }
  }
  const crossCategoryEdges = [...crossCounts.entries()].map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('\u0000');
    return { fromType, toType, edgeType, count };
  });
  const interGroupImports = [...interCounts.entries()].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return { from, to, count };
  });

  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const edge of importEdges) {
      const from = groupById[edge.source];
      const to = groupById[edge.target];
      if (from === group || to === group) totalEdges += 1;
      if (from === group && to === group) internalEdges += 1;
    }
    intraGroupDensity[group] = { internalEdges, totalEdges, density: totalEdges ? internalEdges / totalEdges : 0 };
  }

  const patternTable = new Map();
  const addPatterns = (names, label) => names.forEach((name) => patternTable.set(name, label));
  addPatterns(['routes', 'api', 'controllers', 'endpoints', 'handlers', 'controller', 'routers', 'blueprints', 'serializers'], 'api');
  addPatterns(['services', 'core', 'lib', 'domain', 'logic', 'internal', 'composables', 'mailers', 'jobs', 'channels', 'signals'], 'service');
  addPatterns(['models', 'model', 'db', 'data', 'persistence', 'repository', 'entities', 'entity', 'migrations', 'sql', 'database', 'schema'], 'data');
  addPatterns(['components', 'views', 'pages', 'ui', 'layouts', 'screens'], 'ui');
  addPatterns(['middleware', 'plugins', 'interceptors', 'guards'], 'middleware');
  addPatterns(['utils', 'helpers', 'common', 'shared', 'tools', 'pkg', 'templatetags'], 'utility');
  addPatterns(['config', 'constants', 'env', 'settings', 'management', 'commands'], 'config');
  addPatterns(['__tests__', 'test', 'tests', 'spec', 'specs'], 'test');
  addPatterns(['types', 'interfaces', 'schemas', 'contracts', 'dtos', 'dto', 'request', 'response'], 'types');
  addPatterns(['hooks'], 'hooks');
  addPatterns(['store', 'state', 'reducers', 'actions', 'slices'], 'state');
  addPatterns(['assets', 'static', 'public'], 'assets');
  addPatterns(['cmd', 'bin'], 'entry');
  addPatterns(['docs', 'documentation', 'wiki'], 'documentation');
  addPatterns(['deploy', 'deployment', 'infra', 'infrastructure', 'k8s', 'kubernetes', 'helm', 'charts', 'terraform', 'tf', 'docker'], 'infrastructure');
  addPatterns(['.github', '.gitlab', '.circleci'], 'ci-cd');
  const patternMatches = {};
  for (const group of Object.keys(directoryGroups)) {
    patternMatches[group] = patternTable.get(group.toLowerCase()) || null;
  }

  const lowerPaths = fileNodes.map((node) => node.filePath.replace(/\\/g, '/').toLowerCase());
  const infraFiles = fileNodes.filter((node) => ['infrastructure', 'ci-cd'].includes(filePattern(node.filePath))).map((node) => node.filePath);
  const deploymentTopology = {
    hasDockerfile: lowerPaths.some((p) => /(^|\/)dockerfile([^/]*)$/.test(p)),
    hasCompose: lowerPaths.some((p) => /(^|\/)docker-compose[^/]*\.ya?ml$/.test(p)),
    hasK8s: lowerPaths.some((p) => /(^|\/)(k8s|kubernetes|helm|charts)\//.test(p)),
    hasTerraform: lowerPaths.some((p) => /\.tf(vars)?$/.test(p)),
    hasCI: lowerPaths.some((p) => p.startsWith('.github/workflows/') || p === '.gitlab-ci.yml' || /(^|\/)jenkinsfile$/.test(p)),
    infraFiles,
  };

  const isDataModel = (node) => /(^|\/)(models?|entities|datasets?|data)(\/|$)/i.test(node.filePath) || (node.tags || []).some((tag) => ['model', 'dataset', 'data'].includes(String(tag).toLowerCase()));
  const dataPipeline = {
    schemaFiles: fileNodes.filter((node) => /\.(sql|graphql|gql|proto|prisma)$/i.test(node.filePath) || node.type === 'schema').map((node) => node.filePath),
    migrationFiles: fileNodes.filter((node) => /(^|\/)migrations?\//i.test(node.filePath)).map((node) => node.filePath),
    dataModelFiles: fileNodes.filter(isDataModel).map((node) => node.filePath),
    apiHandlerFiles: fileNodes.filter((node) => /(^|\/)(routes|api|controllers|handlers|endpoints)\//i.test(node.filePath) || node.type === 'endpoint').map((node) => node.filePath),
  };

  const docNodes = fileNodes.filter((node) => node.type === 'document' || /\.(md|rst)$/i.test(node.filePath));
  const groupsWithDocs = new Set();
  for (const node of docNodes) {
    const ownGroup = groupById[node.id];
    if (ownGroup) groupsWithDocs.add(ownGroup);
    const text = `${node.summary || ''} ${node.filePath}`.toLowerCase();
    for (const group of Object.keys(directoryGroups)) {
      if (group !== 'root' && text.includes(group.toLowerCase())) groupsWithDocs.add(group);
    }
  }
  const allGroups = Object.keys(directoryGroups);
  const docCoverage = {
    groupsWithDocs: groupsWithDocs.size,
    totalGroups: allGroups.length,
    coverageRatio: allGroups.length ? groupsWithDocs.size / allGroups.length : 0,
    undocumentedGroups: allGroups.filter((group) => !groupsWithDocs.has(group)),
  };

  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const { from, to } of interGroupImports) {
    if (from === to) continue;
    const pair = [from, to].sort();
    const pairKey = pair.join('\u0000');
    if (seenPairs.has(pairKey)) continue;
    seenPairs.add(pairKey);
    const ab = interCounts.get(`${pair[0]}\u0000${pair[1]}`) || 0;
    const ba = interCounts.get(`${pair[1]}\u0000${pair[0]}`) || 0;
    if (ab === ba) {
      dependencyDirection.push({ dependent: pair[0], dependsOn: pair[1], count: ab, reciprocalCount: ba, tied: true });
    } else if (ab > ba) {
      dependencyDirection.push({ dependent: pair[0], dependsOn: pair[1], count: ab, reciprocalCount: ba });
    } else {
      dependencyDirection.push({ dependent: pair[1], dependsOn: pair[0], count: ba, reciprocalCount: ab });
    }
  }

  const filesPerGroup = Object.fromEntries(Object.entries(directoryGroups).map(([group, ids]) => [group, ids.length]));
  const nodeTypeCounts = Object.fromEntries(Object.entries(nodeTypeGroups).map(([type, ids]) => [type, ids.length]));
  const output = {
    scriptCompleted: true,
    commonPathPrefix: commonPrefixParts.join('/'),
    directoryGroups,
    nodeTypeGroups,
    importAdjacency,
    groupAdjacency: Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, {
      importsFrom: [...groupImportsFrom[group]],
      importedBy: [...groupImportedBy[group]],
    }])),
    crossCategoryEdges,
    nonCodeConnections,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats: { totalFileNodes: fileNodes.length, filesPerGroup, nodeTypeCounts },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
} catch (error) {
  fail(error.stack || error.message || String(error));
}
