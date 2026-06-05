package baiTap1;

public class guessANumber extends Thread {
	private int number;

	public guessANumber(int number) {
		this.number = number;
	}

	public void run() {
		int counter = 0;
		int guess = 0;

		do {
			guess = (int) (Math.random() * 100 + 1);
			System.out.println(this.getName() + " guesses " + guess);
		} while (guess != number);
		System.out.println("** correct! " + this.getName() + " in " + counter + " guesses.**");
	}
}
