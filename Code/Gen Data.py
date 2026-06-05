import pandas as pd
import numpy as np
import os
import shutil
import sys
import findspark

# ==========================================
# 1. CẤU HÌNH MÔI TRƯỜNG (BẮT BUỘC)
# ==========================================
SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
# Bạn dùng Java 11 hay 21 đều được cho file này (vì chạy Local)
# Nhưng tốt nhất dùng Java 11 nếu đã cài
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'
# Hoặc nếu chưa cài Java 11 thì dùng tạm Java 21 cũ:
# JAVA_LOCATION  = r'C:\Zulu\zulu-21'

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

# ==========================================
# 2. CHƯƠNG TRÌNH TẠO DỮ LIỆU
# ==========================================
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification

# Đường dẫn lưu data
DATA_PATH = "C:/DataTest"

# Xóa thư mục cũ nếu có để tạo lại cho sạch
if os.path.exists(DATA_PATH):
    try:
        shutil.rmtree(DATA_PATH)
    except OSError:
        print("Khong xoa duoc thu muc cu, dang ghi de...")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Khởi tạo Spark Local (Chỉ dùng 1 core để tạo data cho nhẹ máy)
print("Dang khoi tao Spark...")
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("GenData") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

SIZES = [100000, 200000, 500000, 1000000, 2000000, 3000000, 4000000]  # Các mốc dữ liệu

print(f"\n=== DANG TAO DU LIEU VAO: {DATA_PATH} ===")

for size in SIZES:
    print(f"\n...Dang xu ly mốc {size} dòng...")

    # 1. Tạo bằng Pandas (Scikit-Learn)
    # Lưu ý: n_redundant=0 để tránh lỗi toán học
    X, y = make_classification(n_samples=size, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    pdf = pd.DataFrame(X.astype(np.float32), columns=['f0', 'f1', 'f2', 'f3'])
    pdf['label'] = y

    # 2. Chuyển sang Spark
    sdf = spark.createDataFrame(pdf)

    # 3. Chia nhỏ file (Repartition)
    # Chia thành 24 phần để sau này 3 máy Worker (mỗi máy 8 core) đọc cho lẹ
    sdf = sdf.repartition(24)

    # 4. Lưu Parquet
    save_path = f"{DATA_PATH}/data_{size}"
    sdf.write.mode("overwrite").parquet(save_path)
    print(f"   -> Da luu xong: {save_path}")

spark.stop()
print("\n" + "=" * 40)
print("XONG! HAY COPY THU MUC C:/DataTest CHO CAC MAY WORKER!")
print("=" * 40)