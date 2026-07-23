const fs = require('fs');
const path = require('path');
const root = process.cwd();
const ua = path.join(root, '.understand-anything');
const batches = JSON.parse(fs.readFileSync(path.join(ua, 'intermediate', 'batches.json'), 'utf8')).batches;

function filePurpose(p) {
  const n = path.basename(p);
  if (p.startsWith('tests/') || /^test/.test(n)) return `Bộ kiểm thử xác minh hành vi và các điều kiện biên của ${n.replace(/\.py$/, '')}, giúp ngăn hồi quy trong pipeline EmotionCLIP-ReID.`;
  if (p.includes('processor')) return `Điều phối vòng lặp huấn luyện/đánh giá trong ${n}, bao gồm luồng dữ liệu, tối ưu hóa, ghi metric và checkpoint.`;
  if (p.includes('loss/')) return `Cài đặt các hàm mất mát trong ${n} để tối ưu biểu diễn, phân loại và độ phân tách mẫu của mô hình.`;
  if (p.includes('solver/')) return `Cung cấp cơ chế optimizer hoặc learning-rate scheduler trong ${n} cho các giai đoạn huấn luyện.`;
  if (p.includes('datasets/')) return `Xử lý dữ liệu trong ${n}, từ đọc manifest/ảnh và biến đổi đến đóng gói batch cho mô hình.`;
  if (p.includes('model/')) return `Định nghĩa thành phần mô hình trong ${n}, kết nối CLIP, prompt, adapter, routing và các đầu ra cảm xúc/ReID.`;
  if (p.includes('metrics')) return `Tính toán metric đánh giá trong ${n}, tổng hợp chất lượng dự đoán và độ tin cậy của mô hình.`;
  if (p.includes('notebook')) return `Cung cấp helper notebook trong ${n} để chạy pipeline, theo dõi tiến độ và xử lý landmark ổn định.`;
  if (p.includes('config/')) return `Định nghĩa và kiểm tra cấu hình trong ${n}, cung cấp mặc định an toàn cho pipeline EmotionCLIP.`;
  if (n.startsWith('train')) return `Điểm vào huấn luyện ${n}, lắp ghép cấu hình, dữ liệu, mô hình, optimizer và các stage training.`;
  if (n.startsWith('infer') || n.startsWith('test')) return `Điểm vào suy luận/đánh giá ${n}, nạp checkpoint và chạy pipeline trên tập dữ liệu đã cấu hình.`;
  return `Cung cấp các thành phần hỗ trợ trong ${n} cho pipeline EmotionCLIP-ReID.`;
}
function tagsFor(p, kind, name='') {
  if (kind === 'class') return ['lop-mo-hinh', 'thanh-phan', 'emotionclip'];
  if (kind === 'function') {
    const t=['ham-nghiep-vu','xu-ly','emotionclip'];
    if (/loss/i.test(name)) t[1]='ham-mat-mat';
    if (/train|stage/i.test(name)) t[1]='huan-luyen';
    if (/metric|eval/i.test(name)) t[1]='danh-gia';
    return t;
  }
  if (p.startsWith('tests/') || /^test/.test(path.basename(p))) return ['kiem-thu','hoi-quy','xac-minh'];
  if (p.includes('processor') || path.basename(p).startsWith('train')) return ['huan-luyen','dieu-phoi','entry-point'];
  if (p.includes('loss/')) return ['ham-mat-mat','toi-uu-hoa','hoc-bieu-dien'];
  if (p.includes('datasets/')) return ['du-lieu','data-pipeline','tien-xu-ly'];
  if (p.includes('model/')) return ['mo-hinh','clip','deep-learning'];
  if (p.includes('solver/')) return ['toi-uu-hoa','scheduler','huan-luyen'];
  return ['utility','emotionclip','pipeline'];
}
function complexity(lines, structure=0) { return lines > 200 || structure > 20 ? 'complex' : lines >= 50 ? 'moderate' : 'simple'; }
function symbolSummary(kind, name, p) {
  if (kind === 'class') return `Lớp ${name} đóng gói trạng thái và hành vi liên quan trong ${path.basename(p)}, phục vụ pipeline EmotionCLIP-ReID.`;
  const verb = /^test_/.test(name) ? 'Xác minh' : /loss/i.test(name) ? 'Tính toán hàm mất mát' : /train|stage/i.test(name) ? 'Thực thi logic huấn luyện' : /metric|eval|inference/i.test(name) ? 'Thực thi logic đánh giá' : 'Thực hiện xử lý';
  return `${verb} ${name} trong mô-đun ${path.basename(p)}, với đầu vào/đầu ra phục vụ luồng xử lý chính.`;
}

for (const b of batches.filter(x => x.batchIndex >= 1 && x.batchIndex <= 5)) {
  const ex = JSON.parse(fs.readFileSync(path.join(ua,'tmp',`ua-file-extract-results-${b.batchIndex}.json`),'utf8'));
  const nodes=[], edges=[];
  for (const r of ex.results) {
    const fid=`file:${r.path}`;
    nodes.push({id:fid,type:'file',name:path.basename(r.path),filePath:r.path,summary:filePurpose(r.path),tags:tagsFor(r.path,'file'),complexity:complexity(r.nonEmptyLines,(r.functions?.length||0)+(r.classes?.length||0)),languageNotes:'Mã Python tổ chức theo mô-đun, dùng PyTorch cho tensor, mô hình và vòng lặp huấn luyện.'});
    const exported = new Set((r.exports||[]).map(x=>x.name));
    const seen = new Set();
    for (const f of (r.functions||[])) {
      if (seen.has(`f:${f.name}`) || ((f.endLine-f.startLine+1)<10 && !exported.has(f.name))) continue;
      seen.add(`f:${f.name}`); const id=`function:${r.path}:${f.name}`;
      nodes.push({id,type:'function',name:f.name,filePath:r.path,lineRange:[f.startLine,f.endLine],summary:symbolSummary('function',f.name,r.path),tags:tagsFor(r.path,'function',f.name),complexity:complexity(f.endLine-f.startLine+1)});
      edges.push({source:fid,target:id,type:'contains',direction:'forward',weight:1.0});
      if(exported.has(f.name)) edges.push({source:fid,target:id,type:'exports',direction:'forward',weight:0.8});
    }
    for (const c of (r.classes||[])) {
      if (seen.has(`c:${c.name}`) || (!exported.has(c.name) && (c.endLine-c.startLine+1)<20 && (c.methods?.length||0)<2)) continue;
      seen.add(`c:${c.name}`); const id=`class:${r.path}:${c.name}`;
      nodes.push({id,type:'class',name:c.name,filePath:r.path,lineRange:[c.startLine,c.endLine],summary:symbolSummary('class',c.name,r.path),tags:tagsFor(r.path,'class',c.name),complexity:complexity(c.endLine-c.startLine+1,c.methods?.length||0)});
      edges.push({source:fid,target:id,type:'contains',direction:'forward',weight:1.0});
      if(exported.has(c.name)) edges.push({source:fid,target:id,type:'exports',direction:'forward',weight:0.8});
    }
    for (const target of (b.batchImportData[r.path]||[])) edges.push({source:fid,target:`file:${target}`,type:'imports',direction:'forward',weight:0.7});
  }
  const parts=Math.ceil(Math.max(nodes.length/60,edges.length/120));
  const sorted=[...b.files].sort((a,z)=>a.path.localeCompare(z.path));
  const chunk=Math.ceil(sorted.length/parts);
  for(let k=0;k<parts;k++){
    const paths=new Set(sorted.slice(k*chunk,(k+1)*chunk).map(x=>x.path));
    const pn=nodes.filter(n=>paths.has(n.filePath)); const ids=new Set(pn.map(n=>n.id));
    const pe=edges.filter(e=>ids.has(e.source));
    fs.writeFileSync(path.join(ua,'intermediate',`batch-${b.batchIndex}${parts>1?`-part-${k+1}`:''}.json`),JSON.stringify({nodes:pn,edges:pe},null,2));
  }
  process.stdout.write(`batch ${b.batchIndex}: ${nodes.length} nodes, ${edges.length} edges, ${parts} parts\n`);
}
