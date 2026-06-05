import os
import time
import shutil

# CẤU HÌNH ĐƯỜNG DẪN
DATA_DIR = r"C:\SparkData"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Nội dung mẫu
base_content = "Hello Spark Big Data Hadoop AI Machine Learning Python Java Scala " * 100


def check_disk_space(required_mb):
    total, used, free = shutil.disk_usage("C:\\")
    free_mb = free // (1024 * 1024)
    print(f">> Dung lượng trống ổ C: {free_mb} MB")

    # Cần dư ra ít nhất 1.5GB cho Win/Spark chạy
    if free_mb < (required_mb + 1500):
        print(f"⚠️ NGUY HIỂM: Bạn cần {required_mb}MB, nhưng ổ cứng sẽ chỉ còn {free_mb - required_mb}MB.")
        print("   -> Spark sẽ không chạy được nếu ổ cứng đầy!")
        return False
    return True


def generate_file(filename, target_size_mb):
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"   [SKIP] File {filename} da ton tai.")
        return filepath

    print(f">> Dang tao file: {filename} (~{target_size_mb} MB)...")
    start_time = time.time()

    with open(filepath, 'w', encoding='utf-8') as f:
        current_size = 0
        target_bytes = target_size_mb * 1024 * 1024
        batch_content = (base_content + "\n") * 1000
        batch_size = len(batch_content)

        while current_size < target_bytes:
            if target_bytes - current_size > batch_size:
                f.write(batch_content)
                current_size += batch_size
            else:
                f.write(base_content + "\n")
                current_size += len(base_content) + 1

    print(f"   [OK] Xong: {filepath} ({time.time() - start_time:.2f}s)")
    return filepath


# === DANH SÁCH FILE (ĐÃ TỐI ƯU CHO 3.7GB) ===
# 1.5GB = 1.5 * 1024 = 1536 MB
files_to_create = [
    ("data_50MB.txt", 50),
    ("data_200MB.txt", 200),
    ("data_500MB.txt", 500),

    ("data_1GB.txt", 1024),

    ("data_1.5GB.txt", 1536)
]

print("=== KIỂM TRA DUNG LƯỢNG Ổ CỨNG ===")
total_size_needed = sum([s for _, s in files_to_create])

if check_disk_space(total_size_needed):
    print(f"\n=== BẮT ĐẦU TẠO DỮ LIỆU (~{total_size_needed} MB) ===")
    for name, size in files_to_create:
        generate_file(name, size)

    print("\n=== HOÀN TẤT ===")
    print(f"Hãy xóa các file cũ không dùng (nếu có) rồi copy '{DATA_DIR}' sang Worker!")
else:
    print("\n❌ DỪNG LẠI: Không đủ chỗ trống an toàn.")