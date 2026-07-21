const fs=require('fs'), path=require('path');
try {
 const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
 const nodes=input.fileNodes, ids=new Set(nodes.map(n=>n.id)), byId=new Map(nodes.map(n=>[n.id,n]));
 const paths=nodes.map(n=>n.filePath||'');
 const groups={}; const groupOf={};
 for(const n of nodes){const p=n.filePath||'';const seg=p.includes('/')?p.split('/')[0]:'root';(groups[seg]??=[]).push(n.id);groupOf[n.id]=seg;}
 const typeGroups={};for(const n of nodes)(typeGroups[n.type]??=[]).push(n.id);
 const fanIn={},fanOut={},pairs={},internal={},total={};for(const n of nodes){fanIn[n.id]=0;fanOut[n.id]=0;}
 for(const e of input.importEdges){if(!ids.has(e.source)||!ids.has(e.target))continue;fanOut[e.source]++;fanIn[e.target]++;const a=groupOf[e.source],b=groupOf[e.target];pairs[`${a}\0${b}`]=(pairs[`${a}\0${b}`]||0)+1;total[a]=(total[a]||0)+1;total[b]=(total[b]||0)+1;if(a===b)internal[a]=(internal[a]||0)+1;}
 const interGroupImports=Object.entries(pairs).filter(([k])=>!k.startsWith(k.split('\0')[0]+'\0'+k.split('\0')[0])).map(([k,count])=>{const [from,to]=k.split('\0');return{from,to,count}});
 const cross={};for(const e of input.allEdges){const a=byId.get(e.source),b=byId.get(e.target);if(!a||!b)continue;const k=`${a.type}\0${b.type}\0${e.type}`;cross[k]=(cross[k]||0)+1;}
 const patternMatches={};const pm={config:'config',datasets:'data',data:'data',model:'service',loss:'service',solver:'service',processor:'service',utils:'utility',tests:'test',docs:'documentation','.github':'ci-cd'};for(const g of Object.keys(groups))if(pm[g])patternMatches[g]=pm[g];
 const infra=paths.filter(p=>/Dockerfile|docker-compose|\.github\/workflows|\.tf$|k8s|kubernetes/i.test(p));
 const docs=new Set(nodes.filter(n=>n.type==='document').map(n=>(n.filePath||'').split('/')[0]||'root'));
 const out={scriptCompleted:true,directoryGroups:groups,nodeTypeGroups:typeGroups,crossCategoryEdges:Object.entries(cross).map(([k,count])=>{const[fromType,toType,edgeType]=k.split('\0');return{fromType,toType,edgeType,count}}),interGroupImports,intraGroupDensity:Object.fromEntries(Object.keys(groups).map(g=>[g,{internalEdges:internal[g]||0,totalEdges:total[g]||0,density:(internal[g]||0)/Math.max(1,total[g]||0)}])),patternMatches,deploymentTopology:{hasDockerfile:infra.some(p=>/Dockerfile/i.test(p)),hasCompose:infra.some(p=>/compose/i.test(p)),hasK8s:infra.some(p=>/k8s|kubernetes/i.test(p)),hasTerraform:infra.some(p=>/\.tf$/i.test(p)),hasCI:infra.some(p=>/workflows/i.test(p)),infraFiles:infra},dataPipeline:{schemaFiles:nodes.filter(n=>n.type==='schema').map(n=>n.filePath),migrationFiles:paths.filter(p=>/migration/i.test(p)),dataModelFiles:paths.filter(p=>/^model\//.test(p)),apiHandlerFiles:[]},docCoverage:{groupsWithDocs:docs.size,totalGroups:Object.keys(groups).length,coverageRatio:docs.size/Math.max(1,Object.keys(groups).length),undocumentedGroups:Object.keys(groups).filter(g=>!docs.has(g))},dependencyDirection:interGroupImports.map(x=>({dependent:x.from,dependsOn:x.to})),fileStats:{totalFileNodes:nodes.length,filesPerGroup:Object.fromEntries(Object.entries(groups).map(([k,v])=>[k,v.length])),nodeTypeCounts:Object.fromEntries(Object.entries(typeGroups).map(([k,v])=>[k,v.length]))},fileFanIn:fanIn,fileFanOut:fanOut};
 fs.writeFileSync(process.argv[3],JSON.stringify(out,null,2));
} catch(e){process.stderr.write(e.stack+'\n');process.exit(1)}
