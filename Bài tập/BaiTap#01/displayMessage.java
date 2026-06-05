package baiTap1;

public class displayMessage implements Runnable {
	private String message;

	public displayMessage(String message) {
		this.message = message;
	}

	public void run() {
		while(true) {
			System.out.println(message);
		}
	}
}
