package baiTap1;

class runnableDemo implements Runnable {
	private Thread t;
	private String threadName;

	runnableDemo(String name) {
		threadName = name;
		System.out.println("Creating " + threadName);
	}

	public void run() {
		System.out.println("Running " + threadName);
		try {
			for (int i = 4; i > 0; i--) {
				System.out.println("Thread: " + threadName + ", " + i);
				Thread.sleep(500);
			}
		} catch (InterruptedException e) {
			System.out.println("Thread " + threadName + "interrupted.");
		}
	}

	public void start() {
		System.out.println("String " + threadName);
		if (t == null) {
			t = new Thread(this, threadName);
			t.start();
		}
	}
}
