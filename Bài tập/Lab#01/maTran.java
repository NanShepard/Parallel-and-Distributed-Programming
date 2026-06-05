package Lab01;

import java.util.Random;
import java.time.format.*;
import java.util.Random;
import java.time.Duration;
import java.time.Instant;

public class maTran {
	public static final int size = 1500;
	private static final int num_threads = 2;

	private static int[][] genMat(int size) {
		Random rand = new Random();
		int[][] matrix = new int[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrix[i][j] = rand.nextInt(10);
			}
		}
		return matrix;
	}

	private static void printMat(int[][] matrix) {
		for (int[] row : matrix) {
			for (int val : row) {
				System.out.printf("%4d", val);
			}
			System.out.println();
		}
	}

	public static void main(String[] args) throws InterruptedException {
		Instant start = Instant.now();

		int[][] a = genMat(size);
		int[][] b = genMat(size);
		int[][] c = new int[size][size];
//		System.out.println("Ma tran 1: ");
//		printMat(a);
//		System.out.println("Ma tran 2: ");
//		printMat(b);

		Thread[] threads = new Thread[num_threads];
		int rowsPerThread = size / num_threads;

		for (int t = 0; t < num_threads; t++) {
			int startRow = t * rowsPerThread;
			int endRow = (t == num_threads - 1) ? size : startRow + rowsPerThread;
			threads[t] = new Thread(new nhanHaiMaTran(a, b, c, startRow, endRow));
			threads[t].start();
		}
		for (int t = 0; t < num_threads; t++) {
			threads[t].join();
		}

//		System.out.println("Ket qua ma tran la: ");
//		printMat(c);

		Instant end = Instant.now();
		Duration timeElapsed = Duration.between(start, end);
		System.out.println("Thoi gian thuc hien theo phuong phap song song la: " + timeElapsed.toMillis()+ " milliseconds.");
		int num_threads = Runtime.getRuntime().availableProcessors();
        System.out.println("Số lõi CPU khả dụng: " + num_threads);
	}
}
