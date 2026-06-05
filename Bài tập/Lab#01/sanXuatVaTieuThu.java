package Lab01;

import java.time.LocalTime;
import java.time.format.*;
import java.util.List;
import java.util.ArrayList;
import java.util.random.*;

public class sanXuatVaTieuThu {
	public static final List<Integer> KhoHang = new ArrayList<>();
	public static final int BUFFER_SIZE = 20;

	private String getCurrentTime() {
		return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss:SSSS"));
	}

	private void TT_SanXuat() {
		while (true) {

			int item = (int) (Math.random() * 100);

			synchronized (KhoHang) {
				if (KhoHang.size() < BUFFER_SIZE) {
					KhoHang.add(item);
					System.out.println(getCurrentTime() + " : [PROD] thêm " + item + " | size=" + KhoHang.size());
				} else {

					System.out.println(getCurrentTime() + " : [PROD] kho ĐẦY | size=" + KhoHang.size());
				}
			}

			try {
				Thread.sleep(200); // Producer nhanh hơn
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				break;
			}
		}
	}

	private void TT_TieuThu() {
		while (true) {
			Integer taken = null;

			synchronized (KhoHang) {
				if (!KhoHang.isEmpty()) {
					taken = KhoHang.remove(0);
					System.out.println(getCurrentTime() + " : [CONS] lấy " + taken + " | size=" + KhoHang.size());
				} else {
					System.out.println(getCurrentTime() + " : [CONS] kho RỖNG | size=0");
				}
			}

			try {
				Thread.sleep(400);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				break;
			}
		}
	}

	public void start_SanXuat() {
		Thread thread = new Thread(this::TT_SanXuat, "Producer");
		System.out.println(getCurrentTime() + " : bắt đầu sản xuất.");
		thread.start();
	}

	public void start_TieuThu() {
		Thread thread = new Thread(this::TT_TieuThu, "Consumer");
		System.out.println(getCurrentTime() + " : bắt đầu tiêu thụ.");
		thread.start();
	}

	public static void main(String[] args) {
		sanXuatVaTieuThu xs = new sanXuatVaTieuThu();
		xs.start_SanXuat();

		sanXuatVaTieuThu tt = new sanXuatVaTieuThu();
		tt.start_TieuThu();
	}
}
