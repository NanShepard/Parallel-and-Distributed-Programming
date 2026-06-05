import time
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MASTER_IP = "26.163.90.8"
HDFS_URL = f"hdfs://{MASTER_IP}:9000"
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
# 2. THAM SỐ TEST
# ==========================================
CORE_LEVELS = [16, 20, 24, 30]
DATA_SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]

# Dictionary lưu kết quả: {16: [t1, t2...], 20: [t1, t2...]}
results_matrix = {core: [] for core in CORE_LEVELS}

print(f"\n=== BẮT ĐẦU BENCHMARK TOÀN DIỆN (FULL MATRIX) ===")
print(f"Cores: {CORE_LEVELS}")
print(f"Data:  {DATA_SIZES}")

# ==========================================
# 3. CHẠY VÒNG LẶP TEST
# ==========================================
for core_limit in CORE_LEVELS:
    print(f"\n" + "#" * 60)
    print(f">>> ĐANG TEST CẤU HÌNH: {core_limit} CORES")
    print(f"#" * 60)

    # 3.1. Khởi tạo Spark (Restart session với cấu hình mới)
    if 'spark' in locals(): spark.stop()
    if 'sc' in locals(): sc.stop()
    time.sleep(5)

    conf = SparkConf() \
        .setAppName(f"Benchmark_Full_{core_limit}Cores") \
        .setMaster(f"spark://{MASTER_IP}:7077") \
        .set("spark.driver.bindAddress", "0.0.0.0") \
        .set("spark.hadoop.fs.defaultFS", HDFS_URL) \
        .set("spark.cores.max", str(core_limit)) \
        .set("spark.executor.memory", "4g") \
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")

    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # 3.2. Chạy qua từng mức dữ liệu
    for size in DATA_SIZES:
        print(f"   --- Data: {size} dòng ---")
        file_path = f"{HDFS_DATA_PATH}/data_{size}.parquet"

        try:
            st = time.time()

            # Đọc & Xử lý
            sdf = spark.read.parquet(file_path)
            vec = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
            sdf_vec = vec.transform(sdf).select("features", "label")

            # Train (Giữ nguyên tham số để so sánh công bằng)
            rf = SparkRF(labelCol="label", featuresCol="features", numTrees=30)
            model = rf.fit(sdf_vec)

            duration = time.time() - st
            results_matrix[core_limit].append(duration)

            print(f"      ✅ Done: {duration:.2f}s")

        except Exception as e:
            print(f"      ❌ Lỗi (File chưa có?): {e}")
            results_matrix[core_limit].append(0)

    # Dừng Spark sau khi xong 1 mức Core
    spark.stop()

# ==========================================
# 4. VẼ BIỂU ĐỒ ĐA ĐƯỜNG (MULTI-LINE CHART)
# ==========================================
print("\n=== ĐANG VẼ BIỂU ĐỒ TỔNG HỢP ===")
plt.figure(figsize=(12, 7))

# Định nghĩa màu và kiểu đường cho từng Core
styles = {
    16: ('#ff9999', 'o-'),  # Đỏ nhạt
    20: ('#ffcc99', 's-'),  # Cam
    24: ('#66b3ff', '^-'),  # Xanh dương
    30: ('#2ca02c', 'D-')  # Xanh lá (Mạnh nhất)
}

x_indices = np.arange(len(DATA_SIZES))
x_labels = [f"{s // 1000}K" if s < 1000000 else f"{s // 1000000}M" for s in DATA_SIZES]

for core, timings in results_matrix.items():
    color, marker = styles.get(core, ('black', 'x-'))
    plt.plot(x_indices, timings, marker[1], color=color, linewidth=2, markersize=6, label=f'{core} Cores')

    # Ghi số liệu ở điểm cuối cùng (4M) để dễ so sánh
    if len(timings) > 0 and timings[-1] > 0:
        plt.annotate(f"{timings[-1]:.1f}s", (x_indices[-1], timings[-1]),
                     textcoords="offset points", xytext=(5, 0), ha='left', color=color, fontweight='bold')

plt.title('Thời gian xử lý theo Dữ liệu & Số Core', fontsize=14)
plt.xlabel('Kích thước dữ liệu', fontsize=12)
plt.ylabel('Thời gian (Giây)', fontsize=12)
plt.xticks(x_indices, x_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Benchmark_Full_Matrix_Result.png")
print("✅ ĐÃ XONG! Mở file 'Benchmark_Full_Matrix_Result.png'")
plt.show()