import time
import pandas as pd
import numpy as np
import os
import sys
import findspark
import matplotlib.pyplot as plt
import shutil

# ==========================================
# 1. CẤU HÌNH (QUAN TRỌNG)
# ==========================================
MASTER_IP = "26.163.90.8"  # <--- Kiểm tra lại IP Master
DATA_PATH_BASE = "file:///C:/DataTest"  # file:/// báo cho Spark biết là đọc local

SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
# Dùng Java 11 cho ổn định nhất
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.datasets import make_classification

# Khởi tạo Spark
print("Dang khoi tao Spark Cluster...")
spark = SparkSession.builder \
    .appName("Final_Benchmark_LocalRead") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Các mốc dữ liệu để test
DATA_SIZES = [100000, 500000, 1000000, 2000000, 5000000]
# Nếu máy mạnh, bỏ comment dòng dưới để test mốc 5 triệu
# DATA_SIZES.append(5000000)

results_sk = []
results_sp = []

print(f"\n=== BAT DAU SO SANH (DATA MIRRORING - KHONG QUA MANG) ===")

for size in DATA_SIZES:
    print(f"\n>>> TEST MOC: {size} DONG")

    # --- 1. SKLEARN (Tạo lại trên RAM để đo công bằng) ---
    print("   [Sklearn] Dang tao data tren RAM & Training...")
    # n_redundant=0 để không lỗi
    X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)

    st = time.time()
    # n_jobs=1 để ép chạy 1 nhân (mô phỏng máy đơn)
    clf = SklearnRF(n_estimators=20, n_jobs=1, random_state=42)
    clf.fit(X, y)
    t_sk = time.time() - st
    results_sk.append(t_sk)
    print(f"   -> Sklearn Time: {t_sk:.2f}s")

    # --- 2. SPARK (Đọc từ ổ cứng Worker) ---
    print("   [Spark] Worker tu doc data tu o C: va Training...")
    st = time.time()

    # Đọc file parquet có sẵn
    path = f"{DATA_PATH_BASE}/data_{size}"
    try:
        sdf = spark.read.parquet(path)

        # Feature Engineering
        vec = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
        sdf_vec = vec.transform(sdf)

        # Training
        rf = SparkRF(labelCol="label", featuresCol="features", numTrees=20)
        rf.fit(sdf_vec)

        t_sp = time.time() - st
        results_sp.append(t_sp)
        print(f"   -> Spark Time:   {t_sp:.2f}s")

        # So sánh nhanh
        if t_sp < t_sk:
            print(f"   => 🚀 SPARK NHANH HON {(t_sk / t_sp):.1f} LAN!")
        else:
            print(f"   => Sklearn van nhanh hon (Data chua du lon)")

    except Exception as e:
        print(f"   ❌ LOI: Worker khong tim thay file tai {path}")
        print("   Hay chac chan ban da copy thu muc DataTest sang o C cua cac may Worker!")
        results_sp.append(0)  # Ghi nhận lỗi

spark.stop()

# --- VẼ BIỂU ĐỒ ---
print("\n=== DANG VE BIEU DO... ===")
plt.figure(figsize=(10, 6))
x = np.arange(len(DATA_SIZES))
width = 0.35

plt.bar(x - width / 2, results_sk, width, label='Sklearn (1 Core)', color='#ff9999')
plt.bar(x + width / 2, results_sp, width, label='Spark (Cluster 3 Nodes)', color='#66b3ff')

plt.xticks(x, [f"{s // 1000}K" if s < 1000000 else f"{s // 1000000}M" for s in DATA_SIZES])
plt.ylabel("Thoi gian (s)")
plt.title("Hieu nang: Sklearn vs Spark (Data Locality Optimization)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ghi số liệu lên cột
for i, v in enumerate(results_sk):
    plt.text(i - width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(results_sp):
    if v > 0:
        plt.text(i + width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=9)

plt.savefig("final_victory_chart_IRIS.png")
print("✅ XONG! Mo file 'final_victory_chart.png' de xem chien thang cua Spark!")
plt.show()