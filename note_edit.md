Khởi tạo prompt
      ↓
1A: mở khóa Cc, học context token
      ↓
Kết thúc 1A
      ↓
Đóng băng Cc
      ↓
1B: mở khóa geometry projector + gate
      ↓
Học residual Δ, nhưng không sửa Cc
      ↓
Kết thúc Stage 1: đóng băng toàn bộ prompt hiệu dụng

A photo of a face with
[V0]
[V1 + Δupper]
[V2 + Δmiddle]
[V3 + Δlower]
showing an angry expression.

STAGE 1
Class-level geometry
"hình học trung bình của lớp happy/sad/anger"
          │
          ▼
Điều chỉnh text prompt
          │
          ▼
Tạo semantic prototype tốt hơn

STAGE 2
Instance-level anatomy
"ảnh cụ thể này: mắt có đáng tin không,
miệng có bị che không, vùng nào nên route?"
          │
          ▼
Chọn/routing patch và vùng mặt
          │
          ▼
Fusion global + local + anatomy

Stage 1 geometry = geometry giúp prompt hiểu class
Stage 2 anatomy  = anatomy giúp model xử lý từng ảnh



```mermaid
flowchart TD
    A[Train split] --> B[CLIP Image Encoder<br/>đóng băng]
    B --> C[Cache global image features<br/>và labels]

    A --> D[Landmark / anatomy artifacts]
    D --> E[Thống kê geometry<br/>theo từng lớp cảm xúc]

    C --> F{STAGE1.MODE}

    F -->|base / 1A| G[Học context tokens Cc]
    G --> H[Text encoder đóng băng]
    H --> I[Text prototype mỗi class]

    F -->|geometry / 1B| J[Đóng băng context Cc]
    E --> J
    J --> K[Học geometry residual Δ<br/>projector + gate]
    K --> I

    I --> L[Cosine similarity<br/>image feature × text prototype]
    C --> L
    L --> M[Cross-entropy loss]
    M --> N[Validation]
    N --> O[best_emotionclip_stage1.pth]
    O --> P[stage1_text_descriptors.pth]
```
