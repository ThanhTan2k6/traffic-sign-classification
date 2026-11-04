# HƯỚNG DẪN TẢI & KIỂM TRA TRỌNG SỐ MÔ HÌNH (MODEL WEIGHTS)

> **Lưu ý quan trọng**: Nhằm tuân thủ quy chuẩn repository tinh gọn (<100 MB cho từng file và <50 MB cho toàn bộ release package khi đưa lên GitHub), toàn bộ các tệp trọng số huấn luyện (.pt, .joblib) được lưu trữ tập trung trên một thư mục Google Drive duy nhất và không đính kèm trực tiếp trong git.

---

## 1. Thư mục tải Trọng số trên Google Drive

Toàn bộ 4 tệp mô hình đã được tải lên và chia sẻ tại thư mục Google Drive:
👉 **[Truy cập Thư mục Google Drive Chứa Trọng Số](https://drive.google.com/drive/folders/1-mLWtYmxFlZIdP0luV6cqNNGqBcMPSfg?hl=vi)**

*Sau khi mở link, bạn hãy tải các file về máy và đặt vào vị trí tương ứng theo bảng dưới đây.*

---

## 2. Bảng đối soát tệp trọng số & Mã băm SHA-256

| Tệp trọng số | Kích thước | Mã băm SHA-256 (kiểm tra tính toàn vẹn) | Đường dẫn tương đối cần đặt sau khi tải | Vai trò trong hệ thống |
|---|:---:|:---:|---|---|
| `vn_s_best.pt` | 18.28 MB | `c6dca365c2d64951113b7b1d0d4145b436de864ac1fbc09a3b37f6bd2693fa6b` | `runs/detect/vn_s/weights/best.pt` | Detector YOLO11s 1 lớp finetune trên dataset Việt Nam |
| `cctsdb_s_best.pt` | 18.28 MB | `8c85b5c68f35bd8280a2288af129b566c7e9b6aaf4a66baa182fd017bc98e4cc` | `runs/detect/cctsdb_s/weights/best.pt` | Detector YOLO11s 1 lớp pretrain trên CCTSDB (trước finetune) |
| `stage2_best.joblib` | 6.50 MB | `3d46eb1052f343a38983cecab805371a7606f6c596745d568d2d6e4beb993c0c` | `models/stage2_best.joblib` | Classifier Stage 2 tối ưu (SVM RBF tại S3 + Scaler, 457 dims) |
| `stage1_hsv_vn_best.joblib` | 2.83 MB | `58a0a275cb56faab30d086470a12b8458471996db60c3fc3c75e9d26b27ff57f` | `models/stage1_hsv_vn_best.joblib` | Classifier Stage 1 baseline (SVM RBF tại S0, 44 dims) |

---

## 3. Hướng dẫn kiểm tra tính toàn vẹn (Checksum Verification)

Sau khi tải các tệp về máy, bạn nên kiểm tra mã băm SHA-256 bằng PowerShell để đảm bảo tệp không bị lỗi đường truyền:

```powershell
# Kiểm tra mã băm SHA-256 của từng file bằng PowerShell
Get-FileHash -Algorithm SHA256 runs\detect\vn_s\weights\best.pt
Get-FileHash -Algorithm SHA256 models\stage2_best.joblib
Get-FileHash -Algorithm SHA256 models\stage1_hsv_vn_best.joblib
```

So sánh chuỗi `Hash` in ra ở terminal với giá trị trong bảng trên. Nếu trùng khớp 100%, tệp đã tải trọn vẹn và sẵn sàng để chạy inference/benchmark.

---

## 4. Lệnh tạo thư mục và bố trí tệp nhanh

Bạn có thể chạy các lệnh PowerShell sau từ thư mục gốc của project để tự động tạo cấu trúc thư mục chứa weights:
```powershell
# Tạo thư mục chứa trọng số YOLO detector
New-Item -ItemType Directory -Force -Path runs\detect\vn_s\weights
New-Item -ItemType Directory -Force -Path runs\detect\cctsdb_s\weights

# Tạo thư mục chứa mô hình classifier
New-Item -ItemType Directory -Force -Path models

# Di chuyển các tệp tải về từ Drive vào đúng vị trí:
# Move-Item <đường-dẫn-tải-về>\vn_s_best.pt runs\detect\vn_s\weights\best.pt
# Move-Item <đường-dẫn-tải-về>\cctsdb_s_best.pt runs\detect\cctsdb_s\weights\best.pt
# Move-Item <đường-dẫn-tải-về>\stage2_best.joblib models\stage2_best.joblib
# Move-Item <đường-dẫn-tải-về>\stage1_hsv_vn_best.joblib models\stage1_hsv_vn_best.joblib
```