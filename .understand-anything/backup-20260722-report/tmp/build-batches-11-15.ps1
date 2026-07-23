param()
$root = 'E:\Source\EmotionCLIP-ReID'
$all = Get-Content -Raw "$root\.understand-anything\intermediate\batches.json" | ConvertFrom-Json
function Complexity([int]$n) { if($n -gt 200){'complex'} elseif($n -ge 50){'moderate'} else {'simple'} }
function NodeType($f) { if($f.fileCategory -eq 'config'){'config'} elseif($f.fileCategory -eq 'docs'){'document'} else {'file'} }
function NodeId($f) { $t=NodeType $f; "$t`:$($f.path)" }
function TagsFor($f) {
  if($f.path -like 'outputs/RAF-DB/metrics_epoch_*.json') { return @('configuration','validation-metrics','raf-db','uncertainty') }
  if($f.fileCategory -eq 'docs') { return @('documentation','dataset','training-artifact') }
  if($f.fileCategory -eq 'data') { return @('data-pipeline','training-metrics','experiment-artifact') }
  if($f.fileCategory -eq 'config') { return @('configuration','experiment-metrics','reporting') }
  if($f.path -like '*.ipynb') { return @('notebook','entry-point','experiment-workflow') }
  if($f.path -like '*.drawio') { return @('documentation','architecture','diagram') }
  if($f.path -like '*.xlsx') { return @('research-data','literature-review','spreadsheet') }
  if($f.path -like '*/__init__.py') { return @('entry-point','package','barrel') }
  if($f.path -like 'config/*') { return @('configuration','hyperparameters','experiment-control') }
  if($f.path -like 'loss/*') { return @('metric-learning','loss-function','model-training') }
  if($f.path -eq 'datasets/preprocessing.py') { return @('data-pipeline','augmentation','preprocessing') }
  if($f.path -eq 'utils/notebook_metrics.py') { return @('utility','metrics','visualization','notebook-support') }
  return @('project-file','artifact','analysis')
}
function SummaryFor($f) {
  if($f.path -match 'outputs/RAF-DB/metrics_epoch_(\d+)\.json') { return "Lưu snapshot đánh giá validation RAF-DB tại epoch $($Matches[1]), gồm accuracy, balanced accuracy, macro-F1, F1 theo lớp, confusion matrix, calibration ECE và chỉ số uncertainty-risk. Tệp còn giữ danh sách ảnh và thống kê confidence/uncertainty để truy vết kết quả." }
  switch($f.path) {
    'outputs/fer2013/README_training_logs.md' { return 'Tài liệu mô tả nguồn gốc và cấu trúc các log training FER2013 đã được trích xuất từ notebook, giúp đối chiếu các bảng loss, validation và uncertainty.' }
    'outputs/fer2013/accuracy_uce_summary.json' { return 'Tổng hợp các epoch tốt nhất của FER2013 theo accuracy, balanced accuracy, macro-F1, ECE và UCE, kèm ghi chú về ý nghĩa metric.' }
    'outputs/fer2013/best_metrics_summary.json' { return 'Snapshot đầy đủ của checkpoint FER2013 tốt nhất, gồm confusion matrix, F1 theo lớp, calibration và uncertainty-risk AUC.' }
    'outputs/fer2013/inference_sample_uncertainty.json' { return 'Ghi kết quả inference của một mẫu FER2013 với nhãn dự đoán, phân phối xác suất, uncertainty và độ tương đồng descriptor.' }
    'outputs/fer2013/training_epoch_losses.csv' { return 'Chuỗi loss training FER2013 được tổng hợp theo epoch để theo dõi xu hướng hội tụ ở cấp vòng lặp.' }
    'outputs/fer2013/training_step_losses.csv' { return 'Log loss chi tiết theo từng optimization step của FER2013, phục vụ phát hiện dao động, divergence và NaN trong training.' }
    'outputs/fer2013/uncertainty_summary.json' { return 'Tổng hợp ngắn checkpoint FER2013 tốt nhất và các metric validation, calibration, uncertainty cùng kết quả inference mẫu.' }
    'outputs/fer2013/validation_accuracy.csv' { return 'Lịch sử validation accuracy của FER2013 theo epoch để chọn checkpoint và đánh giá tiến trình học.' }
    'outputs/fer2013/validation_metrics.csv' { return 'Lịch sử metric validation FER2013 theo epoch, bao phủ accuracy, balanced accuracy, macro-F1 và các chỉ số uncertainty/calibration.' }
    'outputs/fer2013/validation_uncertainty.csv' { return 'Bảng uncertainty validation FER2013 rút gọn, dùng làm nguồn đối chiếu cho báo cáo độ tin cậy.' }
    'outputs/report_w4/emotionclip_outputs_summary.csv' { return 'Bảng tóm tắt cực gọn các output chính của EmotionCLIP-ReID cho FER2013 và RAF-DB để đưa vào báo cáo tuần 4.' }
    'outputs/report_w4/emotionclip_outputs_summary.json' { return 'Tổng hợp có cấu trúc kết quả FER2013 và RAF-DB, liên kết các metric, lịch sử training và artifact phục vụ báo cáo tuần 4.' }
    'outputs/report_w4/rafdb_training_history_extracted.csv' { return 'Lịch sử training RAF-DB đã trích xuất theo epoch, dùng để phân tích loss, hội tụ và so sánh với validation.' }
    'outputs/report_w4/rafdb_validation_history_extracted.csv' { return 'Lịch sử validation RAF-DB đã trích xuất theo epoch, phục vụ chọn checkpoint và lập biểu đồ hiệu năng.' }
    'config/defaults_base.py' { return 'Khai báo cấu hình mặc định nền cho pipeline CLIP-ReID truyền thống, bao gồm model, input, dataset, dataloader, solver và test.' }
    'config/defaults.py' { return 'Khai báo cây cấu hình mặc định mở rộng cho EmotionCLIP-ReID, điều khiển prompt learning, anatomy/geometry, routing, uncertainty và lịch training hai giai đoạn.' }
    'datasets/__init__.py' { return 'Đánh dấu thư mục datasets là Python package; hiện không công bố API hoặc khởi tạo registry.' }
    'datasets/keypoint_test.txt' { return 'Danh sách keypoint/annotation quy mô lớn cho tập test ReID, được lưu như artifact dữ liệu đầu vào thay vì tài liệu diễn giải.' }
    'datasets/keypoint_train.txt' { return 'Danh sách keypoint/annotation quy mô lớn cho tập train ReID, cung cấp metadata hình học cho pipeline dữ liệu.' }
    'datasets/preprocessing.py' { return 'Cung cấp augmentation Random Erasing cho ảnh training, thử nhiều vùng chữ nhật ngẫu nhiên và thay bằng màu trung bình chuẩn hóa.' }
    'loss/__init__.py' { return 'Entry point của package loss, nhập các implementation margin-based loss từ arcface để các module khác sử dụng.' }
    'loss/arcface.py' { return 'Cài đặt ArcFace và CircleLoss dạng module PyTorch cho metric learning, chuẩn hóa embedding/trọng số rồi áp dụng angular margin hoặc circle weighting.' }
    'loss/metric_learning.py' { return 'Tập hợp các objective metric learning gồm ContrastiveLoss, CircleLoss, ArcFace, CosFace và AM-Softmax để huấn luyện embedding phân biệt.' }
    'notebooks/emotionclip_reid_jupyterhub_build_landmarks.ipynb' { return 'Notebook JupyterHub điều phối tiền xử lý landmark cho FER2013 và RAF-DB, tạo manifest anatomy, audit và preview có provenance.' }
    'notebooks/emotionclip_reid_jupyterhub_fer2013.ipynb' { return 'Notebook end-to-end cho thí nghiệm EmotionCLIP-ReID trên FER2013, từ kiểm tra môi trường và dữ liệu đến training, đánh giá và xuất biểu đồ.' }
    'notebooks/emotionclip_reid_jupyterhub_rafdb.ipynb' { return 'Notebook end-to-end cho thí nghiệm EmotionCLIP-ReID trên RAF-DB, triển khai protocol train/validation/test kín và thu thập artifact tái lập.' }
    'outputs/019f6089-e841-7052-847b-c9e0c4537c48/emotionclip_reid_papers.xlsx' { return 'Bảng tổng hợp tài liệu nghiên cứu liên quan EmotionCLIP-ReID, hỗ trợ rà soát baseline, dataset và kết quả công bố.' }
    'outputs/report_w4/emotionclip_reid_w4_functional_change_map.drawio' { return 'Sơ đồ Draw.io ánh xạ các thay đổi chức năng của EmotionCLIP-ReID trong báo cáo tuần 4.' }
    'outputs/report_w4/emotionclip_reid_w4_model_architecture.drawio' { return 'Sơ đồ Draw.io mô tả kiến trúc model EmotionCLIP-ReID và quan hệ giữa các nhánh xử lý.' }
    'outputs/report_w4/emotionclip_reid_w4_two_stage_training.drawio' { return 'Sơ đồ chi tiết quy trình training hai giai đoạn, thể hiện chuyển pha prompt/geometry và tối ưu model FER.' }
    'outputs/report_w4/emotionclip_reid_w4_uncertainty_detail.drawio' { return 'Sơ đồ Draw.io giải thích nhánh uncertainty, calibration và tín hiệu reliability của model.' }
    'processor/__init__.py' { return 'Đánh dấu processor là Python package; tệp rỗng không công bố API ở cấp package.' }
    'solver/__init__.py' { return 'Đánh dấu solver là Python package; tệp rỗng không công bố API ở cấp package.' }
    'utils/__init__.py' { return 'Đánh dấu utils là Python package; tệp rỗng không tái xuất utility.' }
    'utils/notebook_metrics.py' { return 'Cung cấp lớp utility chức năng cho notebook: tìm và hợp nhất metric JSON/CSV, đọc lịch training, in tóm tắt và vẽ metric, confusion matrix cùng F1 theo lớp.' }
    default { return 'Artifact của dự án được lập chỉ mục để phục vụ phân tích kiến trúc và kết quả thí nghiệm.' }
  }
}
$functionSummaries = @{
'metric_epoch'='Trích số epoch từ tên tệp metric để sắp xếp và ghép lịch sử.';'unique_paths'='Khử trùng lặp danh sách đường dẫn trong khi giữ nguyên thứ tự ưu tiên.';'_to_float'='Chuyển giá trị metric sang số thực an toàn và trả fallback cho dữ liệu không hợp lệ.';'_read_json'='Đọc payload JSON từ đường dẫn đã chọn.';'_first_existing'='Chọn đường dẫn tồn tại đầu tiên trong danh sách ứng viên.';'_best_epoch_from_uncertainty'='Suy ra epoch checkpoint tốt nhất từ các artifact uncertainty khả dụng.';'_finalize_metric_bundle'='Chuẩn hóa và hoàn thiện bundle metric, nguồn dữ liệu, epoch tốt nhất và đường dẫn kết quả.';'_load_json_metrics'='Nạp chuỗi metric validation từ các snapshot JSON theo epoch.';'_load_csv_metrics'='Nạp metric validation từ CSV và bổ sung summary/uncertainty khi có.';'load_validation_metrics'='Điều phối chiến lược nạp metric, ưu tiên JSON rồi fallback sang các artifact CSV.';'print_validation_summary'='In bản tóm tắt metric và provenance của checkpoint validation đã chọn.';'_finite_xy'='Lọc cặp epoch-metric hữu hạn để tránh làm hỏng biểu đồ.';'_format_epoch_axis'='Định dạng trục epoch nhất quán cho biểu đồ notebook.';'_plot_metric_lines'='Vẽ nhiều chuỗi metric lên một axes với đánh dấu epoch tốt nhất và xử lý dữ liệu rỗng.';'_pick_payload'='Chọn payload metric đại diện từ bundle cho các biểu đồ chi tiết.';'plot_validation_metric_curves'='Tạo và lưu bộ biểu đồ validation cho accuracy, F1, calibration và uncertainty.';'_parse_log_training_history'='Phân tích log văn bản để trích lịch sử metric training theo epoch.';'_parse_csv_training_history'='Đọc và chuẩn hóa lịch sử training từ tệp CSV.';'load_training_history'='Tìm nguồn log/CSV phù hợp và trả lịch sử training chuẩn hóa.';'plot_training_metric_curves'='Vẽ và lưu các đường loss/metric training từ lịch sử đã chuẩn hóa.';'plot_confusion_matrix_and_f1'='Vẽ confusion matrix và F1 theo lớp từ payload checkpoint validation.'
}
$classSummaries=@{
'RandomErasing'='Augmentation ảnh áp dụng Random Erasing theo xác suất, diện tích và aspect ratio cấu hình.';
'ArcFace'='PyTorch module triển khai angular-margin softmax ArcFace trên embedding và nhãn lớp.';
'CircleLoss'='PyTorch module triển khai Circle Loss với margin và scale cho metric learning.';
'ContrastiveLoss'='Tính contrastive loss theo khoảng cách cặp mẫu và margin để kéo gần/đẩy xa embedding.';
'Cosface'='Triển khai CosFace bằng additive cosine margin trước khi scale logits.';
'AMSoftmax'='Triển khai Additive Margin Softmax cho phân loại embedding.'
}
foreach($idx in 11..15){
  $batch=$all.batches|Where-Object batchIndex -eq $idx
  $extract=Get-Content -Raw "$root\.understand-anything\tmp\ua-file-extract-results-$idx.json"|ConvertFrom-Json
  $nodes=[Collections.Generic.List[object]]::new(); $edges=[Collections.Generic.List[object]]::new()
  foreach($f in $batch.files){
    $er=$extract.results|Where-Object path -eq $f.path
    $nodes.Add([ordered]@{id=(NodeId $f);type=(NodeType $f);name=[IO.Path]::GetFileName($f.path);filePath=$f.path;summary=(SummaryFor $f);tags=@(TagsFor $f);complexity=(Complexity ([int]$er.nonEmptyLines))})
    if($f.fileCategory -eq 'code'){
      foreach($c in @($er.classes)){
        if(!$c){continue}; $len=[int]$c.endLine-[int]$c.startLine+1; if($len -lt 20 -and @($c.methods).Count -lt 2){continue}
        $cid="class:$($f.path):$($c.name)"; $cs=if($classSummaries.ContainsKey($c.name)){$classSummaries[$c.name]}else{"Lớp $($c.name) đóng gói logic chính trong $($f.path)."}
        $nodes.Add([ordered]@{id=$cid;type='class';name=$c.name;filePath=$f.path;lineRange=@([int]$c.startLine,[int]$c.endLine);summary=$cs;tags=@('model-component','metric-learning','pytorch');complexity=(Complexity $len)})
        $edges.Add([ordered]@{source="file:$($f.path)";target=$cid;type='contains';direction='forward';weight=1.0}); $edges.Add([ordered]@{source="file:$($f.path)";target=$cid;type='exports';direction='forward';weight=0.8})
      }
      foreach($fn in @($er.functions)){
        if(!$fn){continue}; $len=[int]$fn.endLine-[int]$fn.startLine+1; $exported=@($er.exports|Where-Object name -eq $fn.name).Count -gt 0; if($len -lt 10 -and !$exported){continue}
        $fid="function:$($f.path):$($fn.name)"; $fs=if($functionSummaries.ContainsKey($fn.name)){$functionSummaries[$fn.name]}else{"Hàm $($fn.name) xử lý một bước trong pipeline metric notebook."}
        $nodes.Add([ordered]@{id=$fid;type='function';name=$fn.name;filePath=$f.path;lineRange=@([int]$fn.startLine,[int]$fn.endLine);summary=$fs;tags=@('utility','metrics','notebook-support');complexity=(Complexity $len)})
        $edges.Add([ordered]@{source="file:$($f.path)";target=$fid;type='contains';direction='forward';weight=1.0}); if($exported){$edges.Add([ordered]@{source="file:$($f.path)";target=$fid;type='exports';direction='forward';weight=0.8})}
      }
      foreach($target in @($batch.batchImportData.($f.path))){$edges.Add([ordered]@{source="file:$($f.path)";target="file:$target";type='imports';direction='forward';weight=0.7})}
    }
  }
  $frag=[ordered]@{nodes=$nodes;edges=$edges}; $out="$root\.understand-anything\intermediate\batch-$idx.json"; [IO.File]::WriteAllText($out,($frag|ConvertTo-Json -Depth 100),[Text.UTF8Encoding]::new($false))
}
