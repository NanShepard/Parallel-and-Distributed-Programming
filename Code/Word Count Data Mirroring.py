import time
import os
import sys
import findspark
import matplotlib.pyplot as plt
import gc

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MASTER_IP = "26.163.90.8"
SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'
DATA_DIR = r"C:\SparkData"  # Thư mục chứa data đã mirror

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

# Fix Java Security
os.environ['_JAVA_OPTIONS'] = "--add-opens=java.base/java.lang=ALL-UNNAMED " \
                              "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"

from pyspark.sql import SparkSession

# ==========================================
# 2. KHỞI TẠO SPARK
# ==========================================
print(">> Dang khoi tao Spark Cluster...")
spark = SparkSession.builder \
    .appName("StressTest_FileBased") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
print(">> Spark Sẵn sàng!")

# ==========================================
# 3. DANH SÁCH FILE CẦN TEST
# ==========================================
# Tên file phải khớp với Bước 1
TEST_FILES = [
    ("50MB", "data_50MB.txt"),
    ("200MB", "data_200MB.txt"),
    ("500MB", "data_500MB.txt"),
    ("1GB", "data_1GB.txt"),
    ("1.5GB", "data_1.5GB.txt")
]

results_py = []
results_sp = []
labels = []

print(f"\n=== BẮT ĐẦU STRESS TEST (IO DISK + CPU) ===")

for label, filename in TEST_FILES:
    file_path = os.path.join(DATA_DIR, filename)

    # Kiểm tra file tồn tại trên Master (Driver)
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy {file_path}. Bỏ qua.")
        continue

    labels.append(label)
    print(f"\n>>> TESTING: {label} ({filename})")

    # --- TEST PYTHON (Đọc file từng dòng để tiết kiệm RAM) ---
    print("   [Python] Đang chạy...", end="\r")
    start_py = time.time()
    wc_py = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    # Logic đơn giản để test tốc độ
                    pass
        time_py = time.time() - start_py
        print(f"   ✅ Python Local: {time_py:.4f}s")
    except Exception as e:
        print(f"   ❌ Python Error: {e}")
        time_py = 0
    results_py.append(time_py)

    # --- TEST SPARK (Đọc file phân tán) ---
    print("   [Spark ] Đang chạy...", end="\r")
    start_sp = time.time()
    try:
        # sc.textFile đọc file từ đường dẫn C:\SparkData trên MỌI Worker
        rdd = sc.textFile(file_path, minPartitions=24)

        # Action đếm số từ
        count = rdd.flatMap(lambda line: line.split()).count()

        time_sp = time.time() - start_sp
        print(f"   ✅ Spark Cluster: {time_sp:.4f}s (Count: {count})")
    except Exception as e:
        print(f"   ❌ Spark Error: {e}")
        print("      (Gợi ý: Kiểm tra xem file đã có trên máy Worker chưa?)")
        time_sp = 0
    results_sp.append(time_sp)

    gc.collect()

# ==========================================
# 4. VẼ BIỂU ĐỒ
# ==========================================
print("\n=== ĐANG VẼ BIỂU ĐỒ... ===")
plt.figure(figsize=(10, 6))

plt.plot(labels, results_py, marker='o', label='Python (Local IO)', color='red', linestyle='--')
plt.plot(labels, results_sp, marker='s', label='Spark (Distributed IO)', color='blue', linewidth=2)

plt.xlabel('Dung lượng dữ liệu')
plt.ylabel('Thời gian xử lý (Giây)')
plt.title('Benchmark: Python vs Spark (File-Based Processing)')
plt.legend()
plt.grid(True)

# Ghi chú giá trị
for i, txt in enumerate(results_py):
    plt.annotate(f"{txt:.1f}s", (labels[i], results_py[i]), textcoords="offset points", xytext=(0, 10), ha='center',
                 color='red')
for i, txt in enumerate(results_sp):
    plt.annotate(f"{txt:.1f}s", (labels[i], results_sp[i]), textcoords="offset points", xytext=(0, -15), ha='center',
                 color='blue')

plt.savefig("FileBased_StressTest.png")
print("✅ ĐÃ XONG! Mở file 'FileBased_StressTest.png' để xem kết quả.")
plt.show()

spark.stop()