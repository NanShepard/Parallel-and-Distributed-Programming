import time
import os
import sys
import findspark

# ==========================================
# 1. CẤU HÌNH
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

# Fix lỗi bảo mật Java 11/21
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
os.environ['_JAVA_OPTIONS'] = JAVA_OPTS

from pyspark.sql import SparkSession

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
file_path = r"D:\Tài liệu đại học\Năm 4\Lập trình song song và phân tán\Dự án cuối kỳ\wordcount.txt"

# Tạo file mẫu lớn hơn nếu chưa có
if not os.path.exists(file_path):
    print("-> Đang tạo file mẫu...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        content = "Hello Spark Hello Big Data Python Java Hadoop Cluster AI Machine Learning\n"
        for _ in range(10000):
            f.write(content)

print(f"--- FILE INPUT: {file_path} ---")
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Kích thước file: {file_size_mb:.2f} MB")

# ==========================================
# PHẦN 1: CHẠY BẰNG PYTHON THUẦN (LOCAL MASTER)
# ==========================================
print("\n" + "=" * 50)
print("PHẦN 1: PYTHON LOCAL (CHỈ CHẠY TRÊN MÁY MASTER)")
print("=" * 50)

try:
    start_py = time.time()

    word_counts_py = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split(" ")
            for word in words:
                if word != "":
                    if word in word_counts_py:
                        word_counts_py[word] += 1
                    else:
                        word_counts_py[word] = 1

    end_py = time.time()
    time_py = end_py - start_py

    print(f"✅ SỐ TỪ KHÁC NHAU (UNIQUE): {len(word_counts_py)}")
    print(f"⏱️ THỜI GIAN PYTHON: {time_py:.4f} giây")

    # --- IN KẾT QUẢ PYTHON ---
    print("\n--- Top 10 từ (Python): ---")
    # Sắp xếp dictionary theo value giảm dần
    sorted_py = sorted(word_counts_py.items(), key=lambda item: item[1], reverse=True)
    for word, count in sorted_py[:10]:
        print(f"'{word}': {count}")

except Exception as e:
    print(f"Lỗi Python Local: {e}")
    time_py = 0

# ==========================================
# PHẦN 2: CHẠY BẰNG SPARK (CLUSTER PHÂN TÁN)
# ==========================================
print("\n" + "=" * 50)
print("PHẦN 2: SPARK CLUSTER (PHÂN TÁN TRÊN 1 MÁY WORKER)")
print("=" * 50)

print("Dang khoi tao Spark...")
spark = SparkSession.builder \
    .appName("Benchmark_WordCount_Output") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.local.dir", TEMP_DIR) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

try:
    start_sp = time.time()

    with open(file_path, 'r', encoding='utf-8') as f:
        local_data = f.readlines()

    rdd = sc.parallelize(local_data)

    counts = rdd.flatMap(lambda line: line.strip().split(" ")) \
        .filter(lambda word: word != "") \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)

    output_sp = counts.collect()

    end_sp = time.time()
    time_sp = end_sp - start_sp

    print(f"✅ SỐ TỪ KHÁC NHAU (UNIQUE): {len(output_sp)}")
    print(f"⏱️ THỜI GIAN SPARK: {time_sp:.4f} giây")

    # --- IN KẾT QUẢ SPARK ---
    print("\n--- Top 10 từ (Spark Cluster): ---")
    # Sắp xếp list of tuples theo count giảm dần
    sorted_sp = sorted(output_sp, key=lambda x: x[1], reverse=True)
    for word, count in sorted_sp[:10]:
        print(f"'{word}': {count}")

    # ==========================================
    # SO SÁNH KẾT QUẢ
    # ==========================================
    print("\n" + "*" * 50)
    print("BẢNG SO SÁNH HIỆU NĂNG")
    print("*" * 50)
    print(f"1. Python Local: {time_py:.4f} s")
    print(f"2. Spark Cluster: {time_sp:.4f} s")

    # Kiểm tra tính đúng đắn (kết quả 2 bên phải giống nhau)
    if len(word_counts_py) == len(output_sp):
        print("\n=> KIỂM TRA: Kết quả số lượng từ khớp nhau ✅")
    else:
        print("\n=> KIỂM TRA: Kết quả bị lệch ❌ (Cần xem lại logic tách từ)")

except Exception as e:
    print("Lỗi Spark:", e)

finally:
    spark.stop()