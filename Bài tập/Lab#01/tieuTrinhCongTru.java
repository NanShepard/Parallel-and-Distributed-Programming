package Lab01;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

public class tieuTrinhCongTru {
	private int interations;
	private String message;
	private int delay;
	public static int count;

	public tieuTrinhCongTru(int interations, String message, int delay) {
		this.interations = interations;
		this.message = message;
		this.delay = delay;
	}

	private void TT_Cong() {
		for (int i = 0; i < interations; i++) {
			try {
				count++;
				System.out.println(getCurrentTime() + ":" + message + ":" + count);
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				System.err.println("Tieu trinh Cong da dung.");
			}
		}
	}

	private void TT_Tru() {
		for (int i = 0; i < interations; i++) {
			try {
				count--;
				System.out.println(getCurrentTime() + ":" + message + ":" + count);
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				System.err.println("Tieu trinh Tru da dung.");
			}
		}
	}

	private String getCurrentTime() {
		return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss:SSSS"));
	}

	public void start_TT_Cong() {
		Thread thread = new Thread(this::TT_Cong);
		System.out.println(getCurrentTime() + " :bat dau tieu trinh cong.");
		thread.start();
	}

	public void start_TT_Tru() {
		Thread thread = new Thread(this::TT_Tru);
		System.out.println(getCurrentTime() + " :bat dau tieu trinh tru.");
		thread.start();
	}

	public static void main(String[] args) throws InterruptedException {
		tieuTrinhCongTru cong = new tieuTrinhCongTru(2500, "Gia tri count trong tieu trinh cong", 1000);
		cong.start_TT_Cong();

		tieuTrinhCongTru tru = new tieuTrinhCongTru(2500, "Gia tri count trong tieu trinh tru", 1000);
		tru.start_TT_Tru();

	}

}
