package Lab01;

class nhanHaiMaTran implements Runnable {
	private int[][] a, b, c;
	private int startRow, endRow;

	public nhanHaiMaTran(int[][] a, int[][] b, int[][] c, int startRow, int endRow) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.startRow = startRow;
		this.endRow = endRow;
	}

	public void run() {
		int n = b.length;
		for (int i = startRow; i < endRow; i++) {
			for (int j = 0; j < n; j++) {
				c[i][j] = 0;
				for (int k = 0; k < n; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}
}
