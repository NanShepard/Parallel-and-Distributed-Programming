import time
import pandas as pd
import numpy as np
import os
import sys
import findspark
import matplotlib.pyplot as plt
import gc

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MASTER_IP = "26.163.90.8"  # <--- IP MASTER ĐÚNG

SPARK_LOCATION = r'C:\Spark'
JAVA_LOCATION = r'C:\Zulu\zulu-21'
HADOOP_LOCATION = r'C:\Hadoop'
TEMP_DIR = "C:\\SparkTemp"

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

# --- CHÌA KHÓA JAVA 21 ---
JAVA_OPTS = "--add-opens=java.base/java.lang=ALL-UNNAMED " \
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED " \
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED " \
            "--add-opens=java.base/java.io=ALL-UNNAMED " \
            "--add-opens=java.base/java.net=ALL-UNNAMED " \
            "--add-opens=java.base/java.nio=ALL-UNNAMED " \
            "--add-opens=java.base/java.util=ALL-UNNAMED " \
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED " \
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED " \
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED " \
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED " \
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED " \
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"

# [QUAN TRỌNG] DÒNG MỚI THÊM ĐỂ FIX LỖI JAVA 21 CHẶN ARROW
os.environ['_JAVA_OPTIONS'] = JAVA_OPTS

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.datasets import make_classification

# ==========================================
# 2. TEST TỰ ĐỘNG
# ==========================================
DATA_SIZES = [100000, 500000, 1000000, 2000000]
results_sklearn = []
results_spark = []

# Khởi tạo Spark
print("Dang khoi tao Spark voi Java Options...")
spark = SparkSession.builder \
    .appName("Auto_Benchmark_Chart") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.local.dir", TEMP_DIR) \
    .getOrCreate()

# Bật Arrow
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "24")

print(f"=== BAT DAU TEST TU DONG TREN {len(DATA_SIZES)} MOC DU LIEU ===")

for size in DATA_SIZES:
    print(f"\n\n>>> DANG CHAY MOC: {size} DONG...")

    # 1. Tạo dữ liệu (Fix lỗi Redundant)
    X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    X = X.astype(np.float32)
    pdf = pd.DataFrame(X, columns=['f0', 'f1', 'f2', 'f3'])
    pdf['label'] = y

    # 2. Đo Sklearn
    print(f"   [Sklearn] Training...")
    clf_sk = SklearnRF(n_estimators=20, n_jobs=1, random_state=42)
    st = time.time()
    clf_sk.fit(pdf[['f0', 'f1', 'f2', 'f3']], pdf['label'])
    t_sk = time.time() - st
    results_sklearn.append(t_sk)
    print(f"   -> Done: {t_sk:.2f}s")

    # 3. Đo Spark
    print(f"   [Spark] Uploading & Training...")
    st = time.time()
    # Repartition 24 để chia đều cho 3 máy
    sdf = spark.createDataFrame(pdf).repartition(24)

    assembler = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
    sdf_vec = assembler.transform(sdf)

    rf_spark = SparkRF(labelCol="label", featuresCol="features", numTrees=20)
    rf_spark.fit(sdf_vec)
    t_sp = time.time() - st
    results_spark.append(t_sp)
    print(f"   -> Done: {t_sp:.2f}s")

    # Dọn rác
    del pdf, X, y, sdf, sdf_vec, clf_sk, rf_spark
    gc.collect()

spark.stop()

# ==========================================
# 3. VẼ BIỂU ĐỒ
# ==========================================
print("\n=== DANG VE BIEU DO... ===")

plt.figure(figsize=(10, 6))
x_labels = [f"{x // 1000}K" if x < 1000000 else f"{x // 1000000}M" for x in DATA_SIZES]
x_axis = np.arange(len(DATA_SIZES))
width = 0.35

plt.bar(x_axis - width / 2, results_sklearn, width, label='Scikit-Learn (1 Core)', color='#ff9999')
plt.bar(x_axis + width / 2, results_spark, width, label='Spark Cluster (3 Nodes)', color='#66b3ff')

plt.xlabel('So luong dong (Rows)')
plt.ylabel('Thoi gian (Giay)')
plt.title('Hieu nang: Sklearn vs Spark Cluster (3 Workers)')
plt.xticks(x_axis, x_labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(results_sklearn):
    plt.text(i - width / 2, v + 0.1, f"{v:.1f}s", ha='center', fontsize=9)
for i, v in enumerate(results_spark):
    plt.text(i + width / 2, v + 0.1, f"{v:.1f}s", ha='center', fontsize=9)

plt.savefig("benchmark_result.png")
print("✅ XONG! Mo file 'benchmark_result.png' de xem ket qua.")
plt.show()