package baiTap1;

public class testThread {
	public static void main(String[] args) {
		runnableDemo R1 = new runnableDemo("Thread - 1");
		R1.start();

		runnableDemo R2 = new runnableDemo("Thread - 2");
		R2.start();
	}
}
