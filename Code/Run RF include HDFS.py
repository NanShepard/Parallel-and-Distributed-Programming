import time
import pandas as pd
import numpy as np
import os
import sys
import findspark
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.datasets import make_classification

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
MASTER_IP = "26.163.90.8"
LOCAL_DIR = r"C:\LocalData"
HDFS_URL = f"hdfs://{MASTER_IP}:9000"
HDFS_DIR = f"{HDFS_URL}/DataTest"

SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'D:\hadoop-3.3.6'
JAVA_LOCATION = r'D:\Java'  # Java 8

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = sys.executable

# ==========================================
# 2. KHỞI TẠO SPARK
# ==========================================
print(">> Dang khoi tao Spark...")
spark = SparkSession.builder \
    .appName("Benchmark_7_Levels") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.hadoop.fs.defaultFS", HDFS_URL) \
    .getOrCreate()


# ==========================================
# 3. HÀM CHUẨN BỊ DỮ LIỆU
# ==========================================
def prepare_data(size):
    local_path = os.path.join(LOCAL_DIR, f"data_{size}")
    hdfs_path = f"{HDFS_DIR}/data_{size}.parquet"

    # 1. Sinh Local (Cho Sklearn)
    if not os.path.exists(local_path):
        print(f"   [GEN] Sinh {size} dòng Local (C:)...")
        X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)
        pdf = pd.DataFrame(X.astype(np.float32), columns=['f0', 'f1', 'f2', 'f3'])
        pdf['label'] = y
        pdf.to_parquet(local_path, engine='pyarrow', index=False)
        del pdf, X, y
    else:
        print(f"   [SKIP] Đã có Local: {size}")

    # 2. Upload HDFS (Cho Spark)
    try:
        # Check xem HDFS có chưa
        spark.read.parquet(hdfs_path).take(1)
        print(f"   [SKIP] Đã có HDFS: {size}")
    except:
        print(f"   [UPLOAD] Đẩy {size} dòng lên HDFS...")
        # Đọc từ local master (file:///)
        path_read = f"file:///{local_path}".replace("\\", "/")
        sdf = spark.read.parquet(path_read)
        # Repartition 4 để chia đều file
        sdf.repartition(4).write.mode("overwrite").parquet(hdfs_path)


# ==========================================
# 4. CHẠY TEST (7 MỨC)
# ==========================================
# [CẬP NHẬT] Danh sách đầy đủ các mốc dữ liệu
DATA_SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]

if not os.path.exists(LOCAL_DIR): os.makedirs(LOCAL_DIR)

res_sk_time, res_sp_time = [], []
res_sk_acc, res_sp_acc = [], []

print("\n=== BƯỚC 1: CHUẨN BỊ DỮ LIỆU (Local & HDFS) ===")
for size in DATA_SIZES:
    prepare_data(size)

print("\n=== BƯỚC 2: BẮT ĐẦU ĐUA (SKLEARN LOCAL vs SPARK HDFS) ===")

for size in DATA_SIZES:
    print(f"\n>>> TEST MỐC: {size} DÒNG")
    local_path = os.path.join(LOCAL_DIR, f"data_{size}")
    hdfs_path = f"{HDFS_DIR}/data_{size}.parquet"

    # --- SKLEARN (Local Read + Split + Train) ---
    print("   [Sklearn] Reading -> Splitting -> Training...")
    try:
        st = time.time()
        # 1. Đọc
        pdf = pd.read_parquet(local_path, engine='pyarrow')
        X = pdf[['f0', 'f1', 'f2', 'f3']]
        y = pdf['label']

        # 2. Chia 80-20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train
        clf = SklearnRF(n_estimators=20, n_jobs=1, random_state=42)
        clf.fit(X_train, y_train)

        t_sk = time.time() - st
        res_sk_time.append(t_sk)

        # 4. Accuracy
        acc_sk = accuracy_score(y_test, clf.predict(X_test)) * 100
        res_sk_acc.append(acc_sk)

        print(f"   -> Sklearn: {t_sk:.2f}s | Acc: {acc_sk:.2f}%")
        del pdf, X, y, clf
        gc.collect()
    except Exception as e:
        print(f"   Lỗi Sklearn: {e}")
        res_sk_time.append(0)
        res_sk_acc.append(0)

    # --- SPARK (HDFS Read + RandomSplit + Train) ---
    print("   [Spark ] HDFS Read -> RandomSplit -> Training...")
    try:
        st = time.time()
        # 1. Đọc HDFS
        sdf = spark.read.parquet(hdfs_path)

        # 2. Vectorize
        vec = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
        sdf_vec = vec.transform(sdf).select("features", "label")

        # 3. Chia 80-20
        train_data, test_data = sdf_vec.randomSplit([0.8, 0.2], seed=42)

        # 4. Train
        rf = SparkRF(labelCol="label", featuresCol="features", numTrees=20)
        model = rf.fit(train_data)

        t_sp = time.time() - st
        res_sp_time.append(t_sp)

        # 5. Accuracy
        preds = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        acc_sp = evaluator.evaluate(preds) * 100
        res_sp_acc.append(acc_sp)

        print(f"   -> Spark:   {t_sp:.2f}s | Acc: {acc_sp:.2f}%")
    except Exception as e:
        print(f"   Lỗi Spark: {e}")
        res_sp_time.append(0)
        res_sp_acc.append(0)

spark.stop()

# ==========================================
# 5. VẼ BIỂU ĐỒ KÉP
# ==========================================
print("\n=== ĐANG VẼ BIỂU ĐỒ ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(DATA_SIZES))
width = 0.35
# Nhãn hiển thị gọn (K hoặc M)
labels = [f"{s // 1000}K" if s < 1000000 else f"{s // 1000000}M" for s in DATA_SIZES]

# Chart 1: Time
ax1.bar(x - width / 2, res_sk_time, width, label='Sklearn (Local)', color='#ff9999')
ax1.bar(x + width / 2, res_sp_time, width, label='Spark (HDFS)', color='#66b3ff')
ax1.set_title('Thời gian thực thi (Giây)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(res_sk_time):
    if v > 0: ax1.text(i - width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=8)
for i, v in enumerate(res_sp_time):
    if v > 0: ax1.text(i + width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=8)

# Chart 2: Accuracy
ax2.plot(labels, res_sk_acc, marker='o', label='Sklearn', color='red')
ax2.plot(labels, res_sp_acc, marker='s', label='Spark', color='blue')
ax2.set_title('Độ chính xác (%)')
ax2.set_ylim(80, 100)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("Final_Benchmark_7_Levels.png")
print("✅ DONE! Check 'Final_Benchmark_7_Levels.png'")
plt.show()