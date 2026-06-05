import time
import os
import sys
import findspark

# ==========================================
# 1. CẤU HÌNH (JAVA 11 + FIX LỖI BẢO MẬT)
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

# Cờ bảo mật bắt buộc cho Java 11/21
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
# 2. KHỞI TẠO SPARK
# ==========================================
print("Dang khoi tao Spark (Java 11)...")
spark = SparkSession.builder \
    .appName("WordCount_Advanced") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.local.dir", TEMP_DIR) \
    .getOrCreate()

sc = spark.sparkContext
print(">>> SparkContext OK!")

# --- BẮT ĐẦU BẤM GIỜ ---
start_time = time.time()

try:
    # ==========================================
    # 3. ĐỌC FILE TỪ ĐƯỜNG DẪN TIẾNG VIỆT
    # ==========================================
    file_path = r"D:\Tài liệu đại học\Năm 4\Lập trình song song và phân tán\Dự án cuối kỳ\wordcount.txt"

    # Tạo file mẫu nếu chưa có
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file tại: {file_path}")
        print("-> Đang tạo file mẫu...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Hello Spark Hello Big Data\n")
            f.write("Chạy Spark trên Windows thật là vui\n")
            f.write("Spark Spark Hadoop Cluster\n")
            f.write("Đếm từ đếm từ đếm từ")

    print(f"--- ĐANG ĐỌC FILE LOCAL: {file_path} ---")

    with open(file_path, 'r', encoding='utf-8') as f:
        local_data = f.readlines()

    print(f"-> Đã đọc {len(local_data)} dòng vào RAM Master.")

    # ==========================================
    # 4. PHÂN TÁN VÀ ĐẾM TỪ
    # ==========================================
    print("--- ĐANG PHÂN TÁN DỮ LIỆU SANG CLUSTER ---")
    rdd = sc.parallelize(local_data)

    print("--- ĐANG XỬ LÝ MAP - REDUCE ---")
    # Logic Word Count
    counts = rdd.flatMap(lambda line: line.strip().split(" ")) \
        .filter(lambda word: word != "") \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b)

    # Thu thập kết quả về Master
    output = counts.collect()

    # --- TÍNH TOÁN THỐNG KÊ ---
    end_time = time.time()
    duration = end_time - start_time
    total_unique_words = len(output)

    print("\n" + "=" * 50)
    print("KẾT QUẢ THỐNG KÊ")
    print("=" * 50)

    # In ra số lượng từ khác nhau
    print(f"✅ TỔNG SỐ TỪ KHÁC NHAU (UNIQUE WORDS): {total_unique_words}")
    print(f"⏱️ THỜI GIAN THỰC THI TOÀN BỘ: {duration:.4f} giây")
    print("-" * 50)

    # In chi tiết danh sách từ (Sắp xếp giảm dần)
    sorted_output = sorted(output, key=lambda x: x[1], reverse=True)

    print("Chi tiết tần suất xuất hiện:")
    for word, count in sorted_output:
        print(f" - '{word}': {count} lần")

    print("=" * 50)

except Exception as e:
    print("\n!!! LỖI !!!")
    import traceback

    traceback.print_exc()

finally:
    spark.stop()