import time
import os
import sys
import gc
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. CẤU HÌNH (Dùng lại đường dẫn cũ)
# ==========================================
MASTER_IP = "26.163.90.8"
HDFS_URL = f"hdfs://{MASTER_IP}:9000"
# Dùng thư mục DataTest cũ (dữ liệu 4 cột f0-f3)
HDFS_DATA_PATH = f"{HDFS_URL}/DataTest"

SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'D:\Hadoop\hadoop-3.3.6'
JAVA_LOCATION = r'D:\Java'

os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = sys.executable

# ==========================================
# 2. KHỞI TẠO SPARK LOCAL (Để tải data)
# ==========================================
print(">>> Khởi tạo Spark Local (Chỉ để đọc HDFS)...")
spark = SparkSession.builder \
    .appName("Sklearn_Loader_NormalData") \
    .master("local[4]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.hadoop.fs.defaultFS", HDFS_URL) \
    .getOrCreate()

# ==========================================
# 3. THAM SỐ TEST
# ==========================================
# Mức dữ liệu y hệt bài test Spark
DATA_SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]

# Đếm số nhân CPU thực tế của máy Master
REAL_CORES = psutil.cpu_count(logical=True)
print(f"--- Máy Master có: {REAL_CORES} Logical Cores ---")

# Cấu hình Sklearn
configs = [
    ("Sklearn (1 Core)", 1),
    (f"Sklearn (Full {REAL_CORES} Cores)", -1)  # -1 là dùng hết nhân
]

results = {name: [] for name, _ in configs}

print("\n=== BẮT ĐẦU BENCHMARK SCIKIT-LEARN (DATA 4 FEATURES) ===")

for size in DATA_SIZES:
    print(f"\n" + "-" * 50)
    print(f"DATASET: {size} dòng")
    file_path = f"{HDFS_DATA_PATH}/data_{size}.parquet"

    # --- BƯỚC 1: Tải dữ liệu về RAM ---
    try:
        # Đọc Parquet từ HDFS
        sdf = spark.read.parquet(file_path)

        # Chuyển sang Pandas (Dữ liệu nằm hoàn toàn trên RAM máy Master)
        pdf = sdf.toPandas()

        # Tách Features (f0-f3) và Label
        # Dựa trên code cũ của bạn: inputCols=['f0', 'f1', 'f2', 'f3']
        X = pdf[['f0', 'f1', 'f2', 'f3']]
        y = pdf['label']

        print(f"   [OK] Đã tải về RAM. Shape: {X.shape}")

        # Dọn dẹp
        del sdf
        gc.collect()

    except Exception as e:
        print(f"   ❌ Lỗi tải dữ liệu: {e}")
        # Điền 0 nếu lỗi để vẽ biểu đồ không bị gãy
        for name in results: results[name].append(0)
        continue

    # --- BƯỚC 2: Training với Scikit-Learn ---
    for label, n_jobs in configs:
        print(f"   running {label}...", end="\r")

        # Cấu hình giống SparkRF: 30 cây
        rf = RandomForestClassifier(n_estimators=30, n_jobs=n_jobs, random_state=42)

        st = time.time()
        rf.fit(X, y)
        duration = time.time() - st

        results[label].append(duration)
        print(f"   ✅ {label}: {duration:.2f}s")

    # Dọn RAM Pandas
    del pdf, X, y, rf
    gc.collect()

spark.stop()

# ==========================================
# 4. VẼ BIỂU ĐỒ
# ==========================================
print("\n=== ĐANG VẼ BIỂU ĐỒ... ===")
plt.figure(figsize=(10, 6))

markers = ['o-', 's-']
colors = ['red', 'blue']

x_indices = range(len(DATA_SIZES))
x_labels = [f"{s // 1000}K" if s < 1000000 else f"{s // 1000000}M" for s in DATA_SIZES]

for i, (label, _) in enumerate(configs):
    plt.plot(x_indices, results[label], markers[i], color=colors[i], linewidth=2, label=label)

    # Ghi chú giá trị tại điểm cuối
    if results[label]:
        last_val = results[label][-1]
        if last_val > 0:
            plt.annotate(f"{last_val:.1f}s", (x_indices[-1], last_val),
                         textcoords="offset points", xytext=(0, 10), ha='center', color=colors[i], fontweight='bold')

plt.title('Scikit-Learn: 1 Core vs Full Cores (Dữ liệu 4 Features)', fontsize=14)
plt.xlabel('Kích thước dữ liệu', fontsize=12)
plt.ylabel('Thời gian Training (Giây)', fontsize=12)
plt.xticks(x_indices, x_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Sklearn_Benchmark_NormalData.png")
print("✅ ĐÃ XONG! Mở file 'Sklearn_Benchmark_NormalData.png'")
plt.show()