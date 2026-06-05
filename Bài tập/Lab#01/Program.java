package Lab01;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

public class Program {
	private int interations;
	private String message;
	private int delay;

	public Program(int interations, String mesage, int delay) {
		this.interations = interations;
		this.message = mesage;
		this.delay = delay;
	}

	public void displayMessage() {
		for (int count = 0; count < interations; count++) {
			System.out.println(getCurrentTime() + ":" + message);
			try {
				Thread.sleep(delay);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				System.err.println("Thread was interrupted.");
			}
		}
	}

	private String getCurrentTime() {
		return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss:SSSS"));
	}

	public void start() {
		// way1
		Thread thread = new Thread(this::displayMessage);
		System.out.println(getCurrentTime() + " : Starting new thread.");
		thread.start();

		// way2
		// new Thread(() -> displayMessage()).start();
	}

	public static void main(String[] args) {
		Program example = new Program(5, "A thread example", 500);

		example.start();

		for (int count = 0; count < 13; count++) {
			System.out.println(example.getCurrentTime() + " : Continue processing...");
			try {
				Thread.sleep(2000);
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				System.err.println("Main thread was interrupted");
			}
		}
		System.out.println("Main method completed. Press enter.");
		try {
			System.in.read();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
