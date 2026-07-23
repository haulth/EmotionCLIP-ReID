const fs = require('fs');
const path = require('path');

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

const normalize = (value) => String(value || '').replaceAll('\\\\', '/').replace(/^\.\//, '');
const extension = (p) => path.posix.extname(normalize(p)).toLowerCase();

function commonDirectoryPrefix(paths) {
  if (!paths.length) return [];
  const dirs = paths.map((p) => normalize(p).split('/').slice(0, -1));
  const shortest = Math.min(...dirs.map((parts) => parts.length));
  const prefix = [];
  for (let i = 0; i < shortest; i += 1) {
    const value = dirs[0][i];
    if (!dirs.every((parts) => parts[i] === value)) break;
    prefix.push(value);
  }
  return prefix;
}

function filePattern(p) {
  const value = normalize(p).toLowerCase();
  const base = path.posix.basename(value);
  if (/(^|\/)(test|tests|__tests__|spec|specs)(\/|$)/.test(value) || /(^test_.*\.py$|.*\.test\.|.*\.spec\.|.*_test\.go$|.*test\.java$|.*_spec\.rb$|.*test\.php$|.*tests\.cs$)/.test(base)) return 'test';
  if (base.endsWith('.d.ts')) return 'types';
  if ((base === 'index.ts' || base === 'index.js' || base === '__init__.py') || base === 'manage.py' || base === 'config.ru') return 'entry';
  if (base === 'wsgi.py' || base === 'asgi.py') return 'config';
  if (['cargo.toml', 'go.mod', 'gemfile', 'pom.xml', 'build.gradle', 'composer.json'].includes(base)) return 'config';
  if (base === 'dockerfile' || base.startsWith('docker-compose.')) return 'infrastructure';
  if (base === 'jenkinsfile' || value.startsWith('.github/workflows/') || base === '.gitlab-ci.yml') return 'ci-cd';
  if (base.endsWith('.tf') || base.endsWith('.tfvars') || base === 'makefile') return 'infrastructure';
  if (base.endsWith('.sql')) return 'data';
  if (base.endsWith('.graphql') || base.endsWith('.gql') || base.endsWith('.proto')) return 'types';
  if (base.endsWith('.md') || base.endsWith('.rst')) return 'documentation';
  return null;
}

function directoryPattern(group) {
  const key = group.toLowerCase();
  const patterns = new Map([
    ['routes','api'],['api','api'],['controllers','api'],['endpoints','api'],['handlers','api'],['serializers','api'],['routers','api'],['blueprints','api'],
    ['services','service'],['core','service'],['lib','service'],['domain','service'],['logic','service'],['internal','service'],['composables','service'],['signals','service'],['mailers','service'],['jobs','service'],['channels','service'],
    ['models','data'],['model','data'],['db','data'],['data','data'],['persistence','data'],['repository','data'],['entities','data'],['entity','data'],['migrations','data'],['sql','data'],['database','data'],['schema','data'],
    ['components','ui'],['views','ui'],['pages','ui'],['ui','ui'],['layouts','ui'],['screens','ui'],
    ['middleware','middleware'],['plugins','middleware'],['interceptors','middleware'],['guards','middleware'],
    ['utils','utility'],['helpers','utility'],['common','utility'],['shared','utility'],['tools','utility'],['pkg','utility'],['templatetags','utility'],
    ['config','config'],['configs','config'],['constants','config'],['env','config'],['settings','config'],['management','config'],['commands','config'],
    ['test','test'],['tests','test'],['__tests__','test'],['spec','test'],['specs','test'],
    ['types','types'],['interfaces','types'],['schemas','types'],['contracts','types'],['dtos','types'],['dto','types'],['request','types'],['response','types'],
    ['hooks','hooks'],['store','state'],['state','state'],['reducers','state'],['actions','state'],['slices','state'],
    ['assets','assets'],['static','assets'],['public','assets'],['cmd','entry'],['bin','entry'],
    ['docs','documentation'],['documentation','documentation'],['wiki','documentation'],
    ['deploy','infrastructure'],['deployment','infrastructure'],['infra','infrastructure'],['infrastructure','infrastructure'],['k8s','infrastructure'],['kubernetes','infrastructure'],['helm','infrastructure'],['charts','infrastructure'],['terraform','infrastructure'],['tf','infrastructure'],['docker','infrastructure'],
    ['.github','ci-cd'],['.gitlab','ci-cd'],['.circleci','ci-cd'],
  ]);
  return patterns.get(key) || null;
}

try {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const nodes = input.fileNodes || [];
  const imports = input.importEdges || [];
  const allEdges = input.allEdges || [];
  const ids = new Set(nodes.map((node) => node.id));
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const normalizedPaths = nodes.map((node) => normalize(node.filePath || node.name));
  const commonPrefixParts = commonDirectoryPrefix(normalizedPaths);

  const groupFor = (node) => {
    const p = normalize(node.filePath || node.name);
    const parts = p.split('/').filter(Boolean);
    const offset = commonPrefixParts.length;
    if (parts.length - offset > 1) return parts[offset];
    if (commonPrefixParts.length === 0 && parts.length > 1) return parts[0];
    if (parts.length <= 1) return 'root';
    const fp = filePattern(p);
    return fp || (extension(p).slice(1) || 'root');
  };

  const directoryGroups = {};
  const nodeTypeGroups = {};
  const nodeGroup = new Map();
  for (const node of nodes) {
    const group = groupFor(node);
    nodeGroup.set(node.id, group);
    (directoryGroups[group] ||= []).push(node.id);
    (nodeTypeGroups[node.type] ||= []).push(node.id);
  }

  const fileFanIn = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const fileFanOut = Object.fromEntries(nodes.map((node) => [node.id, 0]));
  const adjacency = Object.fromEntries(nodes.map((node) => [node.id, []]));
  const interCounts = new Map();
  const groupImportsFrom = {};
  const groupImportedBy = {};
  for (const edge of imports) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) continue;
    fileFanOut[edge.source] += 1;
    fileFanIn[edge.target] += 1;
    adjacency[edge.source].push(edge.target);
    const from = nodeGroup.get(edge.source);
    const to = nodeGroup.get(edge.target);
    (groupImportsFrom[from] ||= new Set()).add(to);
    (groupImportedBy[to] ||= new Set()).add(from);
    if (from !== to) interCounts.set(`${from}\u0000${to}`, (interCounts.get(`${from}\u0000${to}`) || 0) + 1);
  }

  const interGroupImports = [...interCounts.entries()].map(([key, count]) => {
    const [from, to] = key.split('\u0000');
    return { from, to, count };
  }).sort((a,b) => b.count - a.count || a.from.localeCompare(b.from) || a.to.localeCompare(b.to));

  const intraGroupDensity = {};
  for (const group of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const edge of imports) {
      const from = nodeGroup.get(edge.source);
      const to = nodeGroup.get(edge.target);
      if (from === group || to === group) totalEdges += 1;
      if (from === group && to === group) internalEdges += 1;
    }
    intraGroupDensity[group] = { internalEdges, totalEdges, density: totalEdges ? internalEdges / totalEdges : 0 };
  }

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
  }).sort((a,b) => b.count - a.count);

  const patternMatches = {};
  for (const [group, nodeIds] of Object.entries(directoryGroups)) {
    const direct = directoryPattern(group);
    const patterns = nodeIds.map((id) => filePattern(nodeById.get(id)?.filePath || '')).filter(Boolean);
    patternMatches[group] = direct || (patterns.length === nodeIds.length && new Set(patterns).size === 1 ? patterns[0] : null);
  }

  const allPaths = normalizedPaths.map((p) => p.toLowerCase());
  const infraFiles = normalizedPaths.filter((p) => {
    const low = p.toLowerCase();
    return filePattern(low) === 'infrastructure' || filePattern(low) === 'ci-cd' || /(^|\/)(k8s|kubernetes|helm|charts|terraform|infra|infrastructure|docker)(\/|$)/.test(low);
  });
  const deploymentTopology = {
    hasDockerfile: allPaths.some((p) => path.posix.basename(p) === 'dockerfile' || path.posix.basename(p).startsWith('dockerfile.')),
    hasCompose: allPaths.some((p) => path.posix.basename(p).startsWith('docker-compose.')),
    hasK8s: allPaths.some((p) => /(^|\/)(k8s|kubernetes|helm|charts)(\/|$)/.test(p)),
    hasTerraform: allPaths.some((p) => /\.tf(vars)?$/.test(p) || /(^|\/)(terraform|tf)(\/|$)/.test(p)),
    hasCI: allPaths.some((p) => filePattern(p) === 'ci-cd'),
    infraFiles,
  };

  const dataPipeline = {
    schemaFiles: normalizedPaths.filter((p) => /\.(sql|graphql|gql|proto|prisma|schema\.json)$/i.test(p) || /(^|\/)(schema|schemas)(\/|$)/i.test(p)),
    migrationFiles: normalizedPaths.filter((p) => /(^|\/)migrations?(\/|$)/i.test(p)),
    dataModelFiles: nodes.filter((n) => {
      const p = normalize(n.filePath).toLowerCase();
      return /(^|\/)(model|models|datasets?|data|entities|repository)(\/|$)/.test(p) || (n.tags || []).some((t) => ['model','data-model','dataset'].includes(String(t).toLowerCase()));
    }).map((n) => normalize(n.filePath)),
    apiHandlerFiles: nodes.filter((n) => /(^|\/)(api|routes|controllers|handlers|endpoints)(\/|$)/i.test(normalize(n.filePath))).map((n) => normalize(n.filePath)),
  };

  const documentNodes = nodes.filter((node) => node.type === 'document');
  const groupsWithDocsSet = new Set();
  for (const doc of documentNodes) {
    const ownGroup = nodeGroup.get(doc.id);
    if (ownGroup && ownGroup !== 'docs' && ownGroup !== 'documentation') groupsWithDocsSet.add(ownGroup);
    for (const edge of allEdges) {
      if (edge.source === doc.id && ids.has(edge.target)) groupsWithDocsSet.add(nodeGroup.get(edge.target));
      if (edge.target === doc.id && ids.has(edge.source)) groupsWithDocsSet.add(nodeGroup.get(edge.source));
    }
  }
  const meaningfulGroups = Object.keys(directoryGroups).filter((g) => g !== 'docs' && g !== 'documentation');
  const documented = meaningfulGroups.filter((g) => groupsWithDocsSet.has(g));
  const docCoverage = {
    groupsWithDocs: documented.length,
    totalGroups: meaningfulGroups.length,
    coverageRatio: meaningfulGroups.length ? documented.length / meaningfulGroups.length : 0,
    undocumentedGroups: meaningfulGroups.filter((g) => !groupsWithDocsSet.has(g)),
  };

  const directions = [];
  const pairs = new Set(interGroupImports.map((x) => [x.from, x.to].sort().join('\u0000')));
  for (const pair of pairs) {
    const [a,b] = pair.split('\u0000');
    const ab = interCounts.get(`${a}\u0000${b}`) || 0;
    const ba = interCounts.get(`${b}\u0000${a}`) || 0;
    if (ab > ba) directions.push({ dependent: a, dependsOn: b, forwardCount: ab, reverseCount: ba });
    else if (ba > ab) directions.push({ dependent: b, dependsOn: a, forwardCount: ba, reverseCount: ab });
    else directions.push({ dependent: a, dependsOn: b, forwardCount: ab, reverseCount: ba, bidirectionalTie: true });
  }

  const results = {
    scriptCompleted: true,
    commonPathPrefix: commonPrefixParts.join('/'),
    directoryGroups,
    nodeTypeGroups,
    importAdjacency: adjacency,
    groupImportsFrom: Object.fromEntries(Object.entries(groupImportsFrom).map(([k,v]) => [k,[...v]])),
    groupImportedBy: Object.fromEntries(Object.entries(groupImportedBy).map(([k,v]) => [k,[...v]])),
    crossCategoryEdges,
    nonCodeConnections,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection: directions,
    fileStats: {
      totalFileNodes: nodes.length,
      filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([k,v]) => [k,v.length])),
      nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([k,v]) => [k,v.length])),
    },
    fileFanIn,
    fileFanOut,
  };
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2) + '\n', 'utf8');
  console.log(`Analyzed ${nodes.length} file nodes across ${Object.keys(directoryGroups).length} groups.`);
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
