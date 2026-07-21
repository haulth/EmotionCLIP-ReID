const fs = require('fs');

try {
  const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const nodes = input.nodes || [];
  const edges = input.edges || [];
  const layers = input.layers || [];
  const byId = new Map(nodes.map(n => [n.id, n]));
  const fanIn = new Map(nodes.map(n => [n.id, 0]));
  const fanOut = new Map(nodes.map(n => [n.id, 0]));
  for (const e of edges) {
    if (fanOut.has(e.source)) fanOut.set(e.source, fanOut.get(e.source) + 1);
    if (fanIn.has(e.target)) fanIn.set(e.target, fanIn.get(e.target) + 1);
  }
  const rank = (map, field) => [...map].map(([id, value]) => ({id, [field]: value, name: byId.get(id)?.name || id}))
    .sort((a,b) => b[field] - a[field] || a.id.localeCompare(b.id)).slice(0,20);
  const outValues = [...fanOut.values()].sort((a,b)=>a-b);
  const inValues = [...fanIn.values()].sort((a,b)=>a-b);
  const out90 = outValues[Math.floor(outValues.length * .9)] || 0;
  const in25 = inValues[Math.floor(inValues.length * .25)] || 0;
  const entryPointCandidates = [];
  const codeNames = /^(index\.(ts|js)|main\.(ts|js|go|py|rs|cpp|c)|app\.(ts|js|py)|server\.(ts|js)|mod\.rs|manage\.py|wsgi\.py|asgi\.py|run\.py|__main__\.py|Application\.java|Main\.java|Program\.cs|config\.ru|index\.php|App\.swift|Application\.kt)$/;
  for (const n of nodes) {
    let score = 0;
    const path = n.filePath || '';
    if (n.type === 'document' && path === 'README.md') score += 5;
    else if (n.type === 'document' && /^[^/]+\.md$/i.test(path)) score += 2;
    if (n.type === 'file') {
      if (codeNames.test(n.name || '')) score += 3;
      if ((path.match(/\//g)||[]).length <= 1) score += 1;
      if ((fanOut.get(n.id)||0) >= out90) score += 1;
      if ((fanIn.get(n.id)||0) <= in25) score += 1;
    }
    if (score) entryPointCandidates.push({id:n.id,score,name:n.name,summary:n.summary});
  }
  entryPointCandidates.sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id));
  const preferred = byId.has('file:train_emotionclip.py') ? 'file:train_emotionclip.py' : entryPointCandidates.find(x=>byId.get(x.id)?.type==='file')?.id;
  const depthMap = {}, order = [], byDepth = {};
  if (preferred) {
    const q=[preferred]; depthMap[preferred]=0;
    while(q.length){ const cur=q.shift(); order.push(cur); const d=depthMap[cur]; (byDepth[d] ||= []).push(cur);
      for(const e of edges){if(e.source===cur && (e.type==='imports'||e.type==='calls') && byId.has(e.target) && depthMap[e.target]===undefined){depthMap[e.target]=d+1;q.push(e.target)}}
    }
  }
  const inv = {documentation:[],infrastructure:[],data:[],config:[]};
  for(const n of nodes){const item={id:n.id,name:n.name,type:n.type,summary:n.summary}; if(n.type==='document')inv.documentation.push(item); else if(['service','pipeline','resource'].includes(n.type))inv.infrastructure.push(item); else if(['table','schema','endpoint'].includes(n.type))inv.data.push(item); else if(n.type==='config')inv.config.push(item)}
  const pairTypes = new Set(['imports','calls']); const directed = new Set(edges.filter(e=>pairTypes.has(e.type)).map(e=>`${e.source}\0${e.target}`)); const pairs=[];
  for(const e of edges){if(pairTypes.has(e.type)&&directed.has(`${e.target}\0${e.source}`)&&e.source<e.target)pairs.push({nodes:[e.source,e.target],edgeCount:edges.filter(x=>[e.source,e.target].includes(x.source)&&[e.source,e.target].includes(x.target)).length})}
  pairs.sort((a,b)=>b.edgeCount-a.edgeCount);
  const nodeSummaryIndex={}; for(const n of nodes)nodeSummaryIndex[n.id]={name:n.name,type:n.type,summary:n.summary};
  const result={scriptCompleted:true,entryPointCandidates:entryPointCandidates.slice(0,5),fanInRanking:rank(fanIn,'fanIn'),fanOutRanking:rank(fanOut,'fanOut'),bfsTraversal:{startNode:preferred,order,depthMap,byDepth},nonCodeFiles:inv,clusters:pairs.slice(0,10),layers:{count:layers.length,list:layers},nodeSummaryIndex,totalNodes:nodes.length,totalEdges:edges.length};
  fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2));
} catch (err) { console.error(err.stack || String(err)); process.exit(1); }
