import time
import pandas as pd
import numpy as np
import os
import sys
import findspark
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.ensemble import RandomForestClassifier as SklearnRF

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MASTER_IP = "26.163.90.8"
LOCAL_DATA_DIR = r"C:\DataTest"  # Thư mục chứa dữ liệu

SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'
# Fix lỗi bảo mật Java
os.environ[
    '_JAVA_OPTIONS'] = "--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED"

# ==========================================
# 2. KHỞI TẠO SPARK
# ==========================================
print(">> Dang khoi tao Spark Cluster...")
spark = SparkSession.builder \
    .appName("Benchmark_Preview_Data") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .getOrCreate()

print(">> Sẵn sàng Benchmark!")

# ==========================================
# 3. CHẠY TEST
# ==========================================
DATA_SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]

time_sk, time_sp = [], []
acc_sk, acc_sp = [], []

for size in DATA_SIZES:
    folder_name = f"data_{size}"
    full_path_win = os.path.join(LOCAL_DATA_DIR, folder_name)
    full_path_spark = f"file:///{LOCAL_DATA_DIR}/{folder_name}".replace("\\", "/")

    print(f"\n>>> TEST MOC: {size} DONG")
    print(f"    Path: {full_path_win}")

    # --- 1. SKLEARN ---
    print("   [Sklearn] Reading & Training...")
    try:
        st = time.time()
        # Đọc folder parquet
        pdf = pd.read_parquet(full_path_win, engine='pyarrow')

        # --- [MỚI] XUẤT 3 DÒNG ĐẦU TIÊN CỦA SKLEARN ---
        print(f"   [Preview Pandas Data]:")
        print(pdf.head(3))
        print("   ------------------------")
        # ----------------------------------------------

        X = pdf[['f0', 'f1', 'f2', 'f3']]
        y = pdf['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SklearnRF(n_estimators=20, n_jobs=1, random_state=42)
        clf.fit(X_train, y_train)

        t_sk = time.time() - st
        time_sk.append(t_sk)

        y_pred = clf.predict(X_test)
        score_sk = accuracy_score(y_test, y_pred) * 100
        acc_sk.append(score_sk)

        print(f"   -> Time: {t_sk:.2f}s | Acc: {score_sk:.2f}%")
        del pdf, X, y, clf

    except Exception as e:
        print(f"   ❌ Lỗi Sklearn: {e}")
        time_sk.append(0)
        acc_sk.append(0)

    # --- 2. SPARK ---
    print("   [Spark ] Cluster Reading & Training...")
    try:
        st = time.time()
        sdf = spark.read.parquet(full_path_spark)

        # --- [MỚI] XUẤT 3 DÒNG ĐẦU TIÊN CỦA SPARK ---
        print(f"   [Preview Spark Data]:")
        sdf.show(3)
        # --------------------------------------------

        vec = VectorAssembler(inputCols=['f0', 'f1', 'f2', 'f3'], outputCol="features")
        sdf_vec = vec.transform(sdf).select("features", "label")

        train_data, test_data = sdf_vec.randomSplit([0.8, 0.2], seed=42)

        rf = SparkRF(labelCol="label", featuresCol="features", numTrees=20)
        model = rf.fit(train_data)

        t_sp = time.time() - st
        time_sp.append(t_sp)

        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        score_sp = evaluator.evaluate(predictions) * 100
        acc_sp.append(score_sp)

        print(f"   -> Time: {t_sp:.2f}s | Acc: {score_sp:.2f}%")

    except Exception as e:
        print(f"   ❌ Lỗi Spark: {e}")
        time_sp.append(0)
        acc_sp.append(0)

spark.stop()

# ==========================================
# 4. VẼ BIỂU ĐỒ
# ==========================================
if len(time_sk) > 0:
    print("\n=== VE BIEU DO... ===")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(DATA_SIZES))
    width = 0.35
    x_labels = [f"{s // 1000}K" for s in DATA_SIZES]

    # Time Chart
    ax1.bar(x - width / 2, time_sk, width, label='Sklearn (1 Core)', color='#ff9999')
    ax1.bar(x + width / 2, time_sp, width, label='Spark (Cluster)', color='#66b3ff')
    ax1.set_title('Thoi gian (Read + Train) [Thap hon la Tot]')
    ax1.set_ylabel('Giay')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(time_sk):
        ax1.text(i - width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(time_sp):
        if v > 0: ax1.text(i + width / 2, v, f"{v:.1f}s", ha='center', va='bottom', fontsize=9)

    # Accuracy Chart
    ax2.bar(x - width / 2, acc_sk, width, label='Sklearn Acc', color='#99ff99')
    ax2.bar(x + width / 2, acc_sp, width, label='Spark Acc', color='#ffcc99')
    ax2.set_title('Do chinh xac [Cao hon la Tot]')
    ax2.set_ylim(80, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(acc_sk):
        ax2.text(i - width / 2, v, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(acc_sp):
        if v > 0: ax2.text(i + width / 2, v, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("Benchmark_Fixed_Result.png")
    print("✅ DONE! Check 'Benchmark_Fixed_Result.png'")
    plt.show()