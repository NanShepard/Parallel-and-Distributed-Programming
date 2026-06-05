import time
import os
import sys
import findspark
import matplotlib.pyplot as plt
import gc  # Garbage Collector để dọn RAM

# ==========================================
# 1. CẤU HÌNH (JAVA 11/21 + FIX BẢO MẬT)
# ==========================================
MASTER_IP = "26.163.90.8"
SPARK_LOCATION = r'C:\Spark'
HADOOP_LOCATION = r'C:\Hadoop'
TEMP_DIR = "C:\\SparkTemp"
JAVA_LOCATION = r'C:\Program Files\Amazon Corretto\jdk11.0.29_7'  # <--- KIỂM TRA LẠI ĐƯỜNG DẪN NÀY

findspark.init(SPARK_LOCATION)
os.environ['SPARK_HOME'] = SPARK_LOCATION
os.environ['HADOOP_HOME'] = HADOOP_LOCATION
os.environ['JAVA_HOME'] = JAVA_LOCATION
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYSPARK_PYTHON'] = 'python'

# Fix lỗi bảo mật Java
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
# 2. KHỞI TẠO SPARK (CHẠY 1 LẦN DUY NHẤT)
# ==========================================
print(">> Dang khoi tao Spark Cluster...")
spark = SparkSession.builder \
    .appName("StressTest_Upscale") \
    .master(f"spark://{MASTER_IP}:7077") \
    .config("spark.cores.max", "24") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.local.dir", TEMP_DIR) \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
print(">> Spark Sẵn sàng!")

# ==========================================
# 3. CHUẨN BỊ DỮ LIỆU GỐC
# ==========================================
base_file = "base_content.txt"
# Tạo nội dung gốc khoảng 100KB
base_content = "Hello Spark Big Data Hadoop AI Machine Learning Python Java Scala " * 10000

# Các mức nhân bản (Upscale)
MULTIPLIERS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

results_py = []
results_sp = []
data_sizes_mb = []

print(f"\n=== BẮT ĐẦU STRESS TEST (TỪ x{MULTIPLIERS[0]} ĐẾN x{MULTIPLIERS[-1]}) ===")

try:
    for m in MULTIPLIERS:
        # --- BƯỚC 1: TẠO DỮ LIỆU GIẢ LẬP TRÊN RAM ---
        # Nhân bản nội dung lên m lần
        # Lưu ý: Ở mức 5000, chuỗi này có thể lên tới 500MB - 1GB
        current_data = [base_content] * m

        # Tính kích thước ước lượng
        size_mb = (len(base_content) * m) / (1024 * 1024)
        data_sizes_mb.append(size_mb)

        print(f"\n>>> MỨC ĐỘ: x{m} (Kích thước ~{size_mb:.2f} MB)")

        # --- BƯỚC 2: TEST PYTHON (LOCAL) ---
        print("   [Python] Đang chạy...", end="\r")
        start_py = time.time()

        # Logic đếm từ Python thuần
        wc_py = {}
        for line in current_data:  # Giả lập đọc từng dòng
            for word in line.split():
                if word in wc_py:
                    wc_py[word] += 1
                else:
                    wc_py[word] = 1

        time_py = time.time() - start_py
        results_py.append(time_py)
        print(f"   ✅ Python Local: {time_py:.4f}s")

        # --- BƯỚC 3: TEST SPARK (CLUSTER) ---
        print("   [Spark ] Đang chạy...", end="\r")
        start_sp = time.time()

        # Phân tán dữ liệu từ RAM Driver xuống Worker
        # Lưu ý: Với dữ liệu cực lớn (>2GB), bước này có thể nghẽn mạng
        rdd = sc.parallelize(current_data, numSlices=24)

        counts = rdd.flatMap(lambda line: line.split()) \
            .map(lambda word: (word, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .collect()

        time_sp = time.time() - start_sp
        results_sp.append(time_sp)
        print(f"   ✅ Spark Cluster: {time_sp:.4f}s")

        # So sánh nhanh
        if time_sp < time_py:
            print(f"   => 🏆 SPARK THẮNG (Nhanh hơn {time_py / time_sp:.1f} lần)")
        else:
            print(f"   => Python thắng (Spark chậm do overhead)")

        # Dọn dẹp RAM ngay lập tức
        del current_data, rdd, counts, wc_py
        gc.collect()

except MemoryError:
    print("\n⚠️ CẢNH BÁO: TRÀN RAM (OUT OF MEMORY) Ở MỨC CAO NHẤT!")
    print("Đây chính là giới hạn của xử lý cục bộ.")
except Exception as e:
    print(f"\n❌ Lỗi: {e}")

# ==========================================
# 4. VẼ BIỂU ĐỒ KẾT QUẢ
# ==========================================
print("\n=== ĐANG VẼ BIỂU ĐỒ... ===")
plt.figure(figsize=(12, 6))

plt.plot(MULTIPLIERS, results_py, marker='o', label='Python (1 Core)', color='red', linestyle='--')
plt.plot(MULTIPLIERS, results_sp, marker='s', label='Spark Cluster (3 Nodes)', color='blue', linewidth=2)

plt.xlabel('Mức độ nhân bản dữ liệu (Upscale Factor)')
plt.ylabel('Thời gian xử lý (Giây)')
plt.title(f'Stress Test: Python vs Spark (Data ~{data_sizes_mb[0]:.1f}MB đến ~{data_sizes_mb[-1]:.1f}MB)')
plt.legend()
plt.grid(True)
plt.xscale('log')  # Dùng thang đo Logarit để nhìn rõ các mốc nhỏ và lớn

# Ghi chú các điểm
for i, txt in enumerate(results_py):
    if i % 2 == 0:  # Ghi thưa ra cho đỡ rối
        plt.annotate(f"{txt:.1f}s", (MULTIPLIERS[i], results_py[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', color='red')

for i, txt in enumerate(results_sp):
    if i % 2 == 0:
        plt.annotate(f"{txt:.1f}s", (MULTIPLIERS[i], results_sp[i]), textcoords="offset points", xytext=(0, -15),
                     ha='center', color='blue')

plt.savefig("StressTest_Result.png")
print("✅ ĐÃ XONG! Mở file 'StressTest_Result.png' để xem biểu đồ chiến thắng.")
plt.show()

spark.stop()