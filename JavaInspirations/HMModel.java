/**
 * Created with IntelliJ IDEA.
 * User: dennis
 * Date: 11/23/12
 * Time: 2:28 AM
 * To change this template use File | Settings | File Templates.
 */
public class HMModel {
    private double[][] _A;
    private double[][] _B;
    private double[] _PI;

    private double[][] transA;
    private double[][] transB;
    private double[] transPi;

    final public int N;
    final public int M;

    /**
     *
     * @param A matrix of state transition probabilities
     * @param B matrix of observation probabilities
     * @param pi matrix of initial state distribution
     */
    public HMModel(double[][] A, double[][]B, double[] pi) {
        _A = A;
        _B = B;
        _PI = pi;

        N = A.length;
        M = B[0].length;
        transA = new double[N][N];
        transB = new double[N][M];
        transPi = new double[N];

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                transA[i][j] = Math.log(A[i][j]);
            }
            for (int j = 0; j < M; j++) {
                transB[i][j] = Math.log(B[i][j]);
            }
            transPi[i] = Math.log(pi[i]);
        }
    }

    /**
     * Determines if an HMM model is valid, mainly based upon whether the rows
     * of all the probability matrices add to 1.0.
     *
     * @return Whether or not the HMM model is valid
     */
    public boolean isValid() {
        if (_A == null || _B == null || _PI == null) return false;
        for (double[] arr : _A) {
            if (arr == null) return false;
            if (!isRowStochastic(arr)) return false;
        }

        for (double[] arr : _B) {
            if (arr == null) return false;
            if (!isRowStochastic(arr)) return false;
        }

        if (!isRowStochastic(_PI)) return false;
        return true;
    }


    // Checks if row adds to 1.000 +/- .001 and that for all probabilities x
    // 0 <= x <= 1
    private boolean isRowStochastic(double[] arr) {
        double sum = 0;
        for (double d: arr) {
            sum += d;
            if (d < 0 || d > 1.0 ) return false;
        }
        if (sum > .999 && sum < 1.001) return true;
        return false;
    }

    public double[][] getTransformedA() {
        return transA;
    }

    public double[][] getTransformedB() {
        return transB;
    }

    public double[] getTransformedPi() {
        return transPi;
    }

    public double[][] getA() {
        return _A;
    }

    public double[][] getB() {
        return _B;
    }

    public double[] getPi() {
        return _PI;
    }

    public String toString() {
        int decimalPlaces = 3;
        double multiplier = Math.pow(10, decimalPlaces);
        String string = "Pi \n";
        for (double d : _PI) {
            string = string + ((int)(d * multiplier + .5) / multiplier) + " ";
        }
        string = string + "\n Alpha\n";
        if (_A != null) {
            for (double[] row : _A ) {
                for (double d : row) {
                    string = string + ((int)(d * multiplier + .5) / multiplier) + " ";
                }
                string = string + "\n";
            }
        }
        string = string + "\n Beta\n";
        for (double[] row : _B ) {
            for (double d : row) {
                string = string + ((int)(d * multiplier + .5) / multiplier) + " ";
            }
            string = string + "\n";
        }
        return string;
    }
}
