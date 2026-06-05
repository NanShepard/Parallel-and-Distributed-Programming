import pandas as pd
import numpy as np
import os
import shutil
import sys
import findspark

# ==========================================
# 1. CẤU HÌNH MÔI TRƯỜNG
# ==========================================
SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

# ==========================================
# 2. CHƯƠNG TRÌNH TẠO DỮ LIỆU TỪ IRIS
# ==========================================
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris  # <--- Dùng bộ dữ liệu Iris

# Đường dẫn lưu data
DATA_PATH = "C:/DataTest"

# Xóa thư mục cũ để tạo lại
if os.path.exists(DATA_PATH):
    try:
        shutil.rmtree(DATA_PATH)
    except OSError:
        print("Khong xoa duoc thu muc cu, dang ghi de...")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Khởi tạo Spark
print("Dang khoi tao Spark...")
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("GenIrisData") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Load Iris gốc một lần
iris = load_iris()
X_orig = iris.data.astype(np.float32)  # (150, 4)
y_orig = iris.target.astype(np.int32)  # (150,)

# Thêm mốc 10 triệu dòng vào cuối
SIZES = [100000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 1000000000]

print(f"\n=== DANG TAO DU LIEU VAO: {DATA_PATH} ===")

for size in SIZES:
    print(f"\n...Dang xu ly mốc {size} dòng (Base: Iris)...")

    # 1. Logic Nhân bản (Upscaling)
    # Tính số lần cần lặp lại bộ 150 dòng để đủ size yêu cầu
    n_repeats = int(np.ceil(size / len(y_orig)))

    # Nhân bản dữ liệu
    X = np.tile(X_orig, (n_repeats, 1))[:size]
    y = np.tile(y_orig, n_repeats)[:size]

    # QUAN TRỌNG: Xáo trộn (Shuffle)
    # Nếu không shuffle, dữ liệu sẽ bị xếp lớp (lớp 0 hết rồi đến lớp 1...), train sẽ sai.
    indices = np.arange(size)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 2. Tạo Pandas DataFrame
    pdf = pd.DataFrame(X, columns=['f0', 'f1', 'f2', 'f3'])
    pdf['label'] = y

    print(f"   -> Da tao xong Pandas DF ({len(pdf)} rows). Dang chuyen sang Spark...")

    # 3. Chuyển sang Spark DataFrame
    sdf = spark.createDataFrame(pdf)

    # 4. Chia nhỏ file (Repartition)
    # Với 10 triệu dòng, nên chia nhiều partition hơn để Spark Cluster đọc nhanh
    num_partitions = 24 if size < 5000000 else 48
    sdf = sdf.repartition(num_partitions)

    # 5. Lưu Parquet
    save_path = f"{DATA_PATH}/data_{size}"
    sdf.write.mode("overwrite").parquet(save_path)
    print(f"   -> Da luu xong: {save_path}")

    # Giải phóng RAM Python
    del pdf, X, y, indices
    import gc

    gc.collect()

spark.stop()
print("\n" + "=" * 50)
print("✅ XONG! DU LIEU IRIS ĐÃ ĐƯỢC UPSCALED.")
print(f"📁 Kiểm tra thư mục: {DATA_PATH}")
print("⚠️ LƯU Ý: File 10 triệu dòng sẽ khá nặng (~200-300MB), hãy kiên nhẫn khi copy.")
print("=" * 50)