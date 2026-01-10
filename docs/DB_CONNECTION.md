# Kết nối Database

## Cách 1: Adminer (Web UI) - Khuyến nghị

1. Mở browser: http://localhost:8080

3. Click "Login"

## Cách 2: psql (Command line)

```bash
docker-compose exec postgres psql -U fincap_admin -d fincap_db
```

Hoặc từ máy local (nếu đã cài PostgreSQL client):

```bash
psql -h localhost -p 5432 -U fincap_admin -d fincap_db
```

Password: `npg_GiSPTtrV8e6g_Vps_2026`

### Một số lệnh psql hữu ích:

```sql
-- Xem danh sách bảng
\dt

-- Xem cấu trúc bảng
\d training_examples

-- Query dữ liệu
SELECT * FROM training_examples LIMIT 10;

-- Đếm số lượng
SELECT label, COUNT(*) FROM training_examples WHERE is_active = true GROUP BY label;

-- Thoát
\q
```

## Cách 3: DBeaver / pgAdmin (Desktop App)

### DBeaver
1. Tải tại: https://dbeaver.io/download/
2. Tạo connection:
   - Host: `localhost`
   - Port: `5432`
   - Database: `capy_teacher`
   - Username: `capy`
   - Password: `capy123`

### pgAdmin
1. Tải tại: https://www.pgadmin.org/download/
2. Tạo server với thông tin tương tự

## Cách 4: VS Code Extension

1. Cài extension: "PostgreSQL" hoặc "Database Client"
2. Tạo connection với thông tin trên


