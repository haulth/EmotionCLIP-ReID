const fs=require('fs');
const input=JSON.parse(fs.readFileSync('.understand-anything/tmp/ua-arch-input.json','utf8'));
const defs=[
 ['layer:orchestration','Điểm vào và điều phối huấn luyện','Các entry point và processor lắp ghép cấu hình, dữ liệu, mô hình, vòng lặp train/eval và checkpoint cho EmotionCLIP-ReID.'],
 ['layer:model','Mô hình và biểu diễn','Các kiến trúc CLIP, prompt learner, adapter biểu cảm, anatomy routing và đầu phân loại tạo nên biểu diễn của hệ thống.'],
 ['layer:optimization','Hàm mục tiêu và tối ưu hóa','Các loss, optimizer và learning-rate scheduler kiểm soát mục tiêu học và động lực cập nhật tham số qua hai stage.'],
 ['layer:data','Dữ liệu và tiền xử lý','Dataset, manifest, landmark/anatomy pipeline và công cụ chuyển đổi dữ liệu cung cấp batch đầu vào có kiểm soát chất lượng.'],
 ['layer:utility','Tiện ích dùng chung','Metric, logging, theo dõi tiến độ, reranking và helper dùng xuyên suốt huấn luyện, notebook và đánh giá.'],
 ['layer:config','Cấu hình thí nghiệm','Cấu hình mặc định, YAML thí nghiệm và thiết lập môi trường/pytest xác định biến thể mô hình cùng chế độ chạy tái lập.'],
 ['layer:validation','Kiểm thử và xác nhận','Các test tự động kiểm tra loss, metric, model unit, data pipeline, notebook helper và tính đúng đắn của integration.'],
 ['layer:experiment-artifacts','Thực nghiệm, kết quả và tài liệu','Notebook, tài liệu và output cấu hình/kết quả lưu dấu cách sử dụng cùng bằng chứng từ các lần thực nghiệm.']
].map(([id,name,description])=>({id,name,description,nodeIds:[]}));
const m=new Map(defs.map(x=>[x.id,x]));
function layer(n){const p=(n.filePath||'').replace(/\\/g,'/'),base=p.split('/').pop();
 if(p.startsWith('outputs/')||p.startsWith('notebooks/')||n.type==='document')return'layer:experiment-artifacts';
 if(p.startsWith('tests/'))return'layer:validation';
 if(p.startsWith('model/'))return'layer:model';
 if(p.startsWith('loss/')||p.startsWith('solver/'))return'layer:optimization';
 if(p.startsWith('datasets/')||p.startsWith('tools/'))return'layer:data';
 if(p.startsWith('utils/'))return'layer:utility';
 if(p.startsWith('config/')||p.startsWith('configs/')||n.type==='config')return'layer:config';
 if(p.startsWith('processor/')||/^(train|test|infer).*\.py$/.test(base))return'layer:orchestration';
 return'layer:utility';}
for(const n of input.fileNodes)m.get(layer(n)).nodeIds.push(n.id);
for(const d of defs)d.nodeIds.sort();
fs.writeFileSync('.understand-anything/intermediate/layers.json',JSON.stringify(defs,null,2));
const all=defs.flatMap(x=>x.nodeIds), unique=new Set(all), expected=new Set(input.fileNodes.map(n=>n.id));
if(all.length!==expected.size||unique.size!==all.length||[...expected].some(id=>!unique.has(id)))throw new Error(`Layer validation failed assigned=${all.length} unique=${unique.size} expected=${expected.size}`);
console.log(defs.map(x=>`${x.id}: ${x.nodeIds.length}`).join('\n'));
