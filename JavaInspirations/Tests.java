/**
 * Created with IntelliJ IDEA.
 * User: dennis
 * Date: 11/23/12
 * Time: 2:39 AM
 * To change this template use File | Settings | File Templates.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.TreeSet;

public class Tests {

    public static void main(String[] args) {
        EMLearningCurve("eng.train");
    }


    //Model written in yellow notebook.
    //Viterbi should print (2, 0, 0)
    public static void writtenTest1() {
        double[][] A = new double[][]{{.3, .4, .3},{.3, .5, .2},{.7, .2, .1}};
        double[][] B = new double[][]{{.5, .0, .5},{.6, .3, .1},{.3, .4, .3}};
        double[] pi =  new double[]{.3, .3, .4};
        int[] observations = {1, 0, 2};
        HMModel yellow = new HMModel(A, B, pi);
        test(1, yellow, observations, "2 0 0","0.03537");
    }

    // Model written in yellow notebook.
    // Viterbi should print (1, 0, 0)
    public static void writtenTest2() {
        double[][] A = new double[][]{{.7, .3},{.6, .4}};
        double[][] B = new double[][]{{.0, .3, .7},{.5, .3, .2}};
        double[] pi =  new double[]{.5, .5};
        int[] observations = {0, 2, 1};
        HMModel yellow = new HMModel(A, B, pi);
        test(2, yellow, observations, "1 0 0", "0.0375");
    }

    // Model written in yellow notebook.
    // Viterbi should print  (1, 1)
    public static void writtenTest3() {
        double[][] A = new double[][]{{.4, .6},{.3, .7}};
        double[][] B = new double[][]{{.5, .3, .2},{.1, .3, .6}};
        double[] pi =  new double[]{.5, .5};
        int[] observations = {1, 2};
        HMModel yellow = new HMModel(A, B, pi);
        test(3, yellow, observations, "1 1", "0.1485");
    }

    /*public static void PeTest() {
         ArrayList<ArrayList<Integer>> states = new ArrayList<ArrayList<Integer>>()[new ArrayList<Integer>(){0,  5,  3, 12,  0, 10,  8,  2, 12, 9,  3},
             new ArrayList<Integer>(){3, 12, 12,  0,  1,  9, 12,  8,  4, 1, 12, 5},
             new ArrayList<Integer>(){2,  5,  1,  3, 11,  4,  7,  7,  1, 0},
             new ArrayList<Integer>(){9,  9, 11,  4,  1,  5, 12,  9,  7, 8, 12}};
         ArrayList<ArrayList<Integer>> observations ={{23,  2,  0, 13, 21, 24,  6,  6,  3,  1, 18},
                        {16, 19, 13,  8,  5, 15, 22,  3, 11, 11,  7, 3},
                        {20, 24, 12, 10,  5, 15, 15, 20, 19, 11},
                        {15,  8, 11,  20, 21,  6, 17,  11,  11,  5, 2};

         HMModel hmm = HMMAlgorithms.ParameterEstimation(states, observations);
         System.out.print(hmm.isValid() + "\n" + hmm.toString());
     }*/

    public static void test(int testNumber, HMModel hmm, int[] observations, String expViterbi, String expProb) {
        System.out.println("RESULTS OF TEST " + testNumber);
        System.out.println("Is valid model? " + hmm.isValid());

        int[] bestState = HMMAlgorithms.Viterbi(hmm, observations);
        System.out.print("Viterbi Optimal states are : \t");
        for (int i : bestState) {
            System.out.print(i + " ");
        }
        System.out.println();
        int[] posteriorDecode = HMMAlgorithms.PosteriorDecoding(hmm, observations);
        System.out.print("PD Optimal states are : \t");
        for (int i : posteriorDecode) {
            System.out.print(i + " ");
        }

        System.out.println();
        System.out.println("Expected states : \t\t" + expViterbi);
        System.out.println("Probability of Obs. Common Method : \t" + HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations));
        System.out.println("Probability of Obs. ForwardBackwards : \t" + HMMAlgorithms.forwardBackward(hmm, observations));
        System.out.println("Expected Probability : \t" +  expProb);
        System.out.println();
    }

    public static void UniqueTest() {
        try {
            Scanner scan = new Scanner(new File("POS.txt"));

            TreeSet<String> uniqueWords = TrainingDataSet.findUniqueWords(scan);
            // unique : Blue
            int counter = 0;
            for (String s : uniqueWords) {
                System.out.println("word " + counter + " : " + s);
                counter++;
            }
            System.out.println(uniqueWords.isEmpty());
        } catch (FileNotFoundException e) {
            System.out.println("File Not Found.");
        }
    }

    public static void EMTest() {
        double[][] A = new double[][]{{.1, .7, .2},{.2, .1, .7},{.7, .2, .1}};
        double[][] B = new double[][]{{.5, .1, .4},{.6, .3, .1},{.3, .4, .3}};
        double[] pi =  new double[]{.3, .3, .4};
        HMModel hmm = new HMModel(A, B, pi);
        int[][] observations = new int[][]{{1,2,0,1},{1,2,2,0,1},{0,1,1,0,2,1,2, 1, 0, 1, 2, 2, 0}, {2,0,0,2}};

        System.out.println(HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations));

        for (int i = 0; i < 20; i++) {
            hmm = HMMAlgorithms.expectationMaximization(hmm, observations, 1);
            System.out.println(hmm.toString());
            System.out.println("Improved probability : " + HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations));
            System.out.println(hmm.isValid());
        }
    }

    public static void EMTest2() {
        double[][] A = new double[][]{{.1, .7, .2},{.2, .1, .7},{.7, .2, .1}};
        double[][] B = new double[][]{{.5, .1, .4},{.6, .3, .1},{.3, .4, .3}};
        double[] pi =  new double[]{.3, .3, .4};
        HMModel hmm = new HMModel(A, B, pi);
        int[][] observations = new int[][]{{1,2,0,1},{1,2,2,0,1},{0,1,1,0,2,1,2, 1, 0, 1, 2, 2, 0}, {2,0,0,2}};

        System.out.println(HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations));

        for (int i = 0; i < 100; i++) {
            hmm = HMMAlgorithms.expectationMaximization2(hmm, observations, 1);
            System.out.println(hmm.toString());
            System.out.println("Improved probability : " + HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations));
            System.out.println(hmm.isValid());
        }
    }

    public static void ForwardBackwardTest() {
        double[][] A = new double[][]{{.3, .4, .3},{.3, .5, .2},{.7, .2, .1}};
        double[][] B = new double[][]{{.5, .1, .4},{.6, .3, .1},{.3, .4, .3}};
        double[] pi =  new double[]{.3, .3, .4};
        HMModel hmm = new HMModel(A, B, pi);
        int[][] observations = new int[][]{{0,0,2,0},{1,2,2,1,0},{1,1,1,2,2,0}};

        for (int i = 0; i < 10; i++) {
            double[][] alpha = HMMAlgorithms.computeAlpha(hmm, observations[i]);
            double[][] beta =  HMMAlgorithms.computeBeta(hmm, observations[i]);
            System.out.println("Alpha");
            Helper.printMatrix(alpha);
            System.out.println("Beta");
            Helper.printMatrix(beta);
            System.out.println("Marginal");
            Helper.printMatrix(Helper.multiplyMatrix(alpha, beta));
        }
    }

    public static void Underflow() {
        TrainingDataSet tds = new TrainingDataSet("eng.train");
        for (int i : HMMAlgorithms.Viterbi(tds.trainer, new int[]{1652,1048,841,1058,8,3582,2031,19,78,1490,2,36,429,816,3333,69, 5000, 5, 555, 55, 72, 94, 11, 4, 75, 42, 12, 9, 45 ,342 ,65, 767, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 43, 264, 57,658, 543, 884, 2643, 3454})) {
            System.out.print(i + " ");
        }
    }

    public static void LearningCurve(String filename) {
        for (double d = .01; d < .095; d += .01) {
            System.out.println("Percentage of Training Data : " + d);
            TrainingDataSet tds = new TrainingDataSet("eng.train", d);
            tds.test(filename);
        }
        for (double d = .1; d <= 1.0; d += .1) {
            System.out.println("Percentage of Training Data : " + d);
            TrainingDataSet tds = new TrainingDataSet("eng.train", d);
            tds.test(filename);
        }
    }

    public static void EMLearningCurve(String filename) {
        for (double d = .01; d < .095; d += .01) {
            System.out.println("Percentage of Training Data : " + d);
            TrainingDataSet tds = new TrainingDataSet("eng.train", d);
            tds.test(filename);
            tds.EmmisionMaximization(filename, 1);
            tds.test(filename);
        }
        for (double d = .1; d <= 1.0; d += .1) {
            System.out.println("Percentage of Training Data : " + d);
            TrainingDataSet tds = new TrainingDataSet("eng.train", d);
            tds.test(filename);
            tds.EmmisionMaximization(filename, 1);
            tds.test(filename);
        }
    }
}
