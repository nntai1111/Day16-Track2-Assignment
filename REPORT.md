# Lab 16 — Báo cáo kết quả (luồng CPU + LightGBM)

- **Hạ tầng:** Terraform, us-east-1. Gồm VPC, 2 public + 2 private subnet, NAT Gateway, ALB, Bastion (`t3.micro`) và Compute Node (`t3.medium`, 2 vCPU / 4 GB RAM).
- **Dataset:** Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`), 284,807 dòng × 30 feature, tỉ lệ gian lận 0.1727%. File được tải từ bản mirror của TensorFlow (`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`), giống hệt bản trên Kaggle, nên không cần Kaggle API key.
- **Mô hình:** `LGBMClassifier` (lr 0.05, num_leaves 31, max_depth 6, min_child_samples 50, early stopping 100 vòng theo AUC). Chia dữ liệu stratified: train 72% / valid 8% / test 20%.

## Kết quả benchmark

| Metric | Kết quả |
|---|---|
| Thời gian load data | 3.900 s |
| Thời gian training | 5.768 s |
| Best iteration | 74 |
| AUC-ROC | 0.9772 |
| Accuracy | 0.9994 |
| F1-Score | 0.8242 |
| Precision | 0.8929 |
| Recall | 0.7653 |
| Inference latency (1 row) | 1.246 ms (trung vị của 200 lần) |
| Inference throughput (1000 rows) | 3.858 ms / 1000 dòng ≈ 259,209 dòng/s |

Chi tiết xem [`benchmark/benchmark_result.json`](benchmark/benchmark_result.json). Ảnh chụp nằm trong [`screenshots/`](screenshots/).

## Nhận xét

1. **Training rất nhanh trên CPU:** chỉ ~5.8 s cho hơn 200 nghìn dòng trên 2 vCPU. `top` cho thấy LightGBM dùng hết cả 2 core (python3 ~187% CPU). Với dữ liệu dạng bảng cỡ này, không cần GPU.
2. **AUC-ROC 0.977** là mức tốt cho bài toán mất cân bằng nặng. **Accuracy 99.94% không có nhiều ý nghĩa**, vì một model đoán "không gian lận" cho mọi giao dịch cũng đạt 99.83%. Nên dựa vào AUC, F1, Precision và Recall để đánh giá.
3. Ở ngưỡng 0.5, model có Precision 0.89 và Recall 0.77: cảnh báo ít nhầm, nhưng vẫn bỏ sót khoảng 23% giao dịch gian lận. Muốn bắt được nhiều gian lận hơn thì hạ ngưỡng quyết định, đổi lại sẽ có thêm cảnh báo nhầm.
4. **Inference:** dự đoán 1 dòng mất ~1.2 ms, phần lớn là overhead gọi hàm của pandas/sklearn. Dự đoán theo batch đạt ~260 nghìn dòng/s. Như vậy CPU nhỏ đủ để phục vụ real-time scoring.
5. Lúc đầu, cấu hình không có regularization chỉ đạt best_iteration = 1 và AUC 0.928, do early stopping dừng quá sớm vì tập validation chỉ có khoảng 40 mẫu gian lận. Thêm `max_depth`, `min_child_samples` và `reg_lambda` giúp model ổn định hơn.
6. **Chi phí:** cả luồng CPU tốn khoảng **$0.10/giờ**, phần lớn là NAT Gateway (~$0.045/giờ). Chi phí của t3.medium (~$0.042/giờ) gần như không đáng kể so với GPU.
