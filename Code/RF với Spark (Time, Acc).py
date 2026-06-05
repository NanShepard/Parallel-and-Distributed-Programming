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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ==========================================
# 1. CẤU HÌNH (GIỮ NGUYÊN NHƯ CŨ)
# ==========================================
MASTER_IP = "26.163.90.8"

SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
TEMP_DIR = "C:\\SparkTemp"
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

# ==========================================
# 2. KHỞI TẠO SPARK
# ==========================================
print("Dang khoi tao Spark (Java 11)...")
spark = SparkSession.builder \
    .appName("Benchmark_Time_Vs_Accuracy") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.local.dir", TEMP_DIR) \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "24")

# ==========================================
# 3. TEST TỰ ĐỘNG (THỜI GIAN & ĐỘ CHÍNH XÁC)
# ==========================================
DATA_SIZES = [100000, 500000, 1000000, 2000000]

# List lưu kết quả
time_sklearn = []
time_spark = []
acc_sklearn = []
acc_spark = []

print(f"=== BAT DAU TEST (TIME & ACCURACY) ===")

for size in DATA_SIZES:
    print(f"\n>>> DANG CHAY MOC: {size} DONG...")

    # --- 1. Tạo dữ liệu ---
    X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    X = X.astype(np.float32)
    pdf = pd.DataFrame(X, columns=['f0', 'f1', 'f2', 'f3'])
    pdf['label'] = y

    # --- 2. Đo Sklearn ---
    print(f"   [Sklearn] Splitting & Training...")
    # Chia train/test 80-20
    X_train, X_test, y_train, y_test = train_test_split(pdf[['f0', 'f1', 'f2', 'f3']], pdf['label'], test_size=0.2,
                                                        random_state=42)

    clf_sk = SklearnRF(n_estimators=20, n_jobs=1, random_state=42)

    # Đo thời gian train
    st = time.time()
    clf_sk.fit(X_train, y_train)
    t_sk = time.time() - st
    time_sklearn.append(t_sk)

    # Đo độ chính xác
    y_pred = clf_sk.predict(X_test)
    score_sk = accuracy_score(y_test, y_pred)
    acc_sklearn.append(score_sk * 100)  # Lưu dạng %

    print(f"   -> Time: {t_sk:.2f}s | Acc: {score_sk * 100:.2f}%")

    # --- 3. Đo Spark ---
    print(f"   [Spark] Uploading, Splitting & Training...")

    # Tạo DataFrame và Vector hóa
    sdf = spark.createDataFrame(pdf).repartition(24)
    assembler = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
    sdf_vec = assembler.transform(sdf).select("features", "label")

    # Chia train/test 80-20 trên Spark
    train_data, test_data = sdf_vec.randomSplit([0.8, 0.2], seed=42)

    rf_spark = SparkRF(labelCol="label", featuresCol="features", numTrees=20)

    # Đo thời gian train (chỉ tính lúc fit, vì action thực sự xảy ra ở đây)
    st = time.time()
    model = rf_spark.fit(train_data)
    t_sp = time.time() - st
    time_spark.append(t_sp)

    # Đo độ chính xác
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    score_sp = evaluator.evaluate(predictions)
    acc_spark.append(score_sp * 100)  # Lưu dạng %

    print(f"   -> Time: {t_sp:.2f}s | Acc: {score_sp * 100:.2f}%")

    # Dọn dẹp RAM
    del pdf, X, y, sdf, sdf_vec, clf_sk, model, train_data, test_data
    gc.collect()

spark.stop()

# ==========================================
# 4. VẼ 2 BIỂU ĐỒ (THỜI GIAN & ĐỘ CHÍNH XÁC)
# ==========================================
print("\n=== DANG VE BIEU DO SO SANH... ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 hàng, 2 cột

x_labels = [f"{x // 1000}K" if x < 1000000 else f"{x // 1000000}M" for x in DATA_SIZES]
x_axis = np.arange(len(DATA_SIZES))
width = 0.35

# --- Biểu đồ 1: Thời gian ---
ax1.bar(x_axis - width / 2, time_sklearn, width, label='Sklearn (1 Core)', color='#ff9999')
ax1.bar(x_axis + width / 2, time_spark, width, label='Spark (Cluster)', color='#66b3ff')
ax1.set_xlabel('So luong dong')
ax1.set_ylabel('Thoi gian (Giay)')
ax1.set_title('Training Time (Thap hon la Tot hon)')
ax1.set_xticks(x_axis)
ax1.set_xticklabels(x_labels)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Ghi số liệu lên biểu đồ 1
for i, v in enumerate(time_sklearn):
    ax1.text(i - width / 2, v + 0.1, f"{v:.1f}s", ha='center', fontsize=9)
for i, v in enumerate(time_spark):
    ax1.text(i + width / 2, v + 0.1, f"{v:.1f}s", ha='center', fontsize=9)

# --- Biểu đồ 2: Độ chính xác ---
ax2.bar(x_axis - width / 2, acc_sklearn, width, label='Sklearn', color='#99ff99')
ax2.bar(x_axis + width / 2, acc_spark, width, label='Spark', color='#ffcc99')
ax2.set_xlabel('So luong dong')
ax2.set_ylabel('Do chinh xac (%)')
ax2.set_title('Accuracy (Cao hon la Tot hon)')
ax2.set_xticks(x_axis)
ax2.set_xticklabels(x_labels)
ax2.legend()
ax2.set_ylim(70, 100)  # Zoom vào khoảng 70-100% để thấy sự chênh lệch nếu có
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Ghi số liệu lên biểu đồ 2
for i, v in enumerate(acc_sklearn):
    ax2.text(i - width / 2, v + 0.2, f"{v:.1f}%", ha='center', fontsize=9)
for i, v in enumerate(acc_spark):
    ax2.text(i + width / 2, v + 0.2, f"{v:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("benchmark_full_time_acc.png")
print("✅ DONE! Check benchmark_full_time_acc.png")
plt.show()