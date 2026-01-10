Tổng kết Chiến lược Triển khai AI (PhoBERT & Gemini)

1. Chiến lược Dữ liệu (Data Flywheel)
Bạn đang xây dựng một "bánh đà dữ liệu" để model tự thông minh lên theo thời gian:
Giai đoạn Dev: Dùng Gemini 2.5 Flash-Lite để dán nhãn dữ liệu thực từ người dùng (Gold Dataset).
Giai đoạn Train: Gom 20k - 60k mẫu (Unique) để fine-tune PhoBERT trên máy RTX 4060.
Giai đoạn Go-live: PhoBERT thay thế Gemini xử lý phần lớn giao dịch, Gemini chỉ đóng vai trò "giám sát" hoặc xử lý câu khó.

2. Kiến trúc Điều hướng (Hybrid Routing)
Để tối ưu chi phí và hiệu năng, hệ thống phân luồng như sau:
Tầng 1 (Keyword): Câu $\le 3$ từ $\rightarrow$ So khớp từ khóa (Free/Instant).
Tầng 2 (Cache): Câu đã có trong DB $\rightarrow$ Trả kết quả cũ (Free/Instant).
Tầng 3 (PhoBERT - Local): Câu 4-10 từ $\rightarrow$ PhoBERT xử lý (Free/Fast).
Nếu độ tự tin (Confidence) $> 0.9$: Trả kết quả.
Nếu độ tự tin thấp hoặc quá tải: Chuyển sang Tầng 4.
Tầng 4 (Gemini - Cloud): Câu $> 10$ từ hoặc ca khó $\rightarrow$ Gọi Gemini API (Tốn phí/Chính xác).

3. Kỹ thuật "Ép cân" Model (Optimization)
Để nhét PhoBERT vào VPS 4GB RAM đang chạy NestJS, Postgres, Redis:
Định dạng: Chuyển từ PyTorch (.bin) sang ONNX.
Nén: Dùng Quantization INT8 (Giảm file từ 540MB $\rightarrow$ 150MB, RAM từ 2GB $\rightarrow$ 300MB).
Môi trường: Trên VPS không cài torch hay transformers. Chỉ dùng onnxruntime, tokenizers (Rust) và underthesea.
Đánh đổi: Chấp nhận giảm ~1% độ chính xác để đổi lấy tốc độ và tiết kiệm RAM 4-5 lần.

4. Quản lý Tài nguyên VPS (Survival Mode)
Chiến thuật để 5-6 dịch vụ không "giẫm chân" nhau trên 4GB RAM:
SWAP: Tạo 2GB Swap file làm phao cứu sinh chống tràn RAM.
Docker Limits: Cấu hình mem_limit cho từng container (Postgres, Redis, NestJS).
Worker: Giới hạn Python Service chỉ chạy 1-2 worker để không làm nghẽn CPU.
Async/Await: Sử dụng lập trình bất đồng bộ để NestJS không bị block khi AI đang tính toán.

5. Kế hoạch thực hiện (Action Plan)
Tại máy Local (RTX 4060):
Fine-tune PhoBERT với 4.5k mẫu hiện có.
Export sang ONNX và nén INT8.
Export tokenizer.json.
Tại VPS:
Tạo Swap file.
Cài đặt Python Service (FastAPI + ONNX Runtime).
Thiết lập chốt chặn Cache (Redis) và Routing logic.
Vận hành:
Mở cho 30-500 anh em vào test.
Lưu lại mọi "Raw text + Label" để tái huấn luyện bản v2, v3.
Thông điệp cuối: Bạn không chỉ đang xây dựng một ứng dụng, bạn đang học cách làm AI thực dụng. Việc tối ưu thành công trên 4GB RAM sẽ là một kinh nghiệm cực kỳ giá trị cho sự nghiệp của bạn.