package Lab01;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

public class nhanMaTranTuanTu {
	private static final int size = 1500;

	// Gen Matrix
	private static int[][] genMat(int size) {
		Random rand = new Random();
		int[][] matrix = new int[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrix[i][j] = rand.nextInt(5);
			}
		}
		return matrix;
	}

	// Print Matrix
	private static void printMat(int[][] matrix) {
		for (int[] row : matrix) {
			for (int val : row) {
				System.out.printf("%4d", val);
			}
			System.out.println();
		}
	}

	public static void main(String[] args) {
		Instant start = Instant.now();
		int[][] a = genMat(size);
		int[][] b = genMat(size);
//		System.out.println("Ma tran 1: ");
//		printMat(a);
//		System.out.println("Ma tran 2: ");
//		printMat(b);
		int[][] c = new int[size][size];
		int n = b.length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < n; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
//		System.out.println("Ket qua ma tran la: ");
//		printMat(c);
		Instant end = Instant.now();
		Duration timeElapsed = Duration.between(start, end);
		System.out.println("Time taken theo phuong phap tuan tu: " + timeElapsed.toMillis() + " milliseconds");

	}

}