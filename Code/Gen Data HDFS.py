import pandas as pd
import numpy as np
import os
import sys
import findspark
import time  # <--- Thư viện đo thời gian

# ==========================================
# 1. CẤU HÌNH
# ==========================================
SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'D:\hadoop-3.3.6'
JAVA_LOCATION = r'D:\Java'

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ==========================================
# 2. CẤU HÌNH KẾT NỐI HDFS
# ==========================================
MASTER_IP = "26.163.90.8"
HDFS_URL = f"hdfs://{MASTER_IP}:9000"
HDFS_DATA_PATH = f"{HDFS_URL}/DataTest"

from pyspark.sql import SparkSession
from sklearn.datasets import make_classification

# Khởi tạo Spark
print("Dang khoi tao Spark...")
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("GenData_To_HDFS_Timed") \
    .config("spark.driver.memory", "4g") \
    .config("spark.hadoop.fs.defaultFS", HDFS_URL) \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
# ^^^ Bật Arrow để tăng tốc chuyển đổi Pandas -> Spark

SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]

# List để lưu kết quả tổng hợp cuối cùng
summary_stats = []

print(f"\n=== DANG TAO DU LIEU VAO HDFS: {HDFS_DATA_PATH} ===")

for size in SIZES:
    print(f"\n" + "-" * 40)
    print(f">>> MỐC DỮ LIỆU: {size} DÒNG")

    # --- GIAI ĐOẠN 1: SINH DỮ LIỆU (CPU & RAM Local) ---
    t_start_gen = time.time()

    X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    pdf = pd.DataFrame(X.astype(np.float32), columns=['f0', 'f1', 'f2', 'f3'])
    pdf['label'] = y

    t_end_gen = time.time()
    duration_gen = t_end_gen - t_start_gen
    print(f"   [1] Sinh xong Pandas DF: {duration_gen:.2f} giây")

    # --- GIAI ĐOẠN 2: UPLOAD & PHÂN TÁN (Network I/O & Disk Write) ---
    t_start_upload = time.time()

    # Bước chuyển đổi Pandas -> Spark tốn thời gian Serialization
    sdf = spark.createDataFrame(pdf)

    # Repartition (Chuẩn bị chia file)
    sdf = sdf.repartition(4)

    # Ghi xuống HDFS (Lúc này dữ liệu mới thực sự chạy qua dây mạng)
    save_path = f"{HDFS_DATA_PATH}/data_{size}.parquet"
    print(f"   [2] Dang upload len HDFS...")
    sdf.write.mode("overwrite").parquet(save_path)

    t_end_upload = time.time()
    duration_upload = t_end_upload - t_start_upload
    print(f"   -> Upload xong: {duration_upload:.2f} giây")

    # Lưu thống kê
    summary_stats.append((size, duration_gen, duration_upload))

    # Dọn RAM
    del pdf, X, y, sdf

spark.stop()

# ==========================================
# 3. BẢNG TỔNG KẾT
# ==========================================
print("\n" + "=" * 60)
print(f"{'SIZE (Dòng)':<15} | {'SINH DATA (s)':<15} | {'UPLOAD HDFS (s)':<15}")
print("-" * 60)
for item in summary_stats:
    print(f"{item[0]:<15} | {item[1]:<15.2f} | {item[2]:<15.2f}")
print("=" * 60)
print("GHI CHÚ:")
print(" - Sinh Data: Tốc độ CPU/RAM của máy chạy code này.")
print(" - Upload HDFS: Tốc độ mạng LAN/VPN + Tốc độ ổ cứng các máy Worker.")