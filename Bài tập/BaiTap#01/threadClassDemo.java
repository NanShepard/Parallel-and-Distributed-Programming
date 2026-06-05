package baiTap1;

public class threadClassDemo {
	public static void main(String[] args) {
		Runnable hello = new displayMessage("Hello");
		Thread thread1 = new Thread(hello);
		thread1.setDaemon(true);
		thread1.setName("Hello");
		System.out.println("Starting hello thread...");
		thread1.start();

		Runnable bye = new displayMessage("Goodbye");
		Thread thread2 = new Thread(bye);
		thread2.setPriority(Thread.MIN_PRIORITY);
		thread2.setDaemon(true);
		System.out.println("Starting goodbye thread...");
		thread2.start();

		System.out.println("Starting thread3...");
		Thread thread3 = new guessANumber(27);
		thread3.start();
		try {
			thread3.join();
		} catch (InterruptedException e) {
			System.out.println("Thread was interrupted.");
		}

		System.out.println("Starting thread4...");
		Thread thread4 = new guessANumber(75);
		thread4.start();

		System.out.println("main() is ending...");
	}
}
