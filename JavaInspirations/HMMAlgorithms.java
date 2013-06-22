/**
 * Created with IntelliJ IDEA.
 * User: dennis
 * Date: 11/23/12
 * Time: 2:31 AM
 * To change this template use File | Settings | File Templates.
 */
public class HMMAlgorithms {

    public static double calculateProbabilityOfObservations(HMModel hmm, int[][] observations) {
        double probability = 1;
        for (int i = 0; i < observations.length; i++) {
            double rowProbability = Math.log(HMMAlgorithms.calculateProbabilityOfObservations(hmm, observations[i]));
            probability += rowProbability;
        }
        return probability;
    }

    /*
     * Given a model and a list of observations, calculates the probability of
     * the list of observations occuring on the model.
     *
     * @param A - State Transition Matrix (N x N)
     * @param B - Emmision Probability Matrix (N x M)
     * @param pi - Intial Distribution Matrix (1 x N)
     * @param observations - list of observations
     * @return the probability that the list of observations would happen
     */


    public static double calculateProbabilityOfObservations(HMModel hmm, int[] observations) {
        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();
        int N = A.length;
        int T = observations.length;

        // Compute a-pass for first step
        double[][] A_Pass = new double[N][T];

        double c = 0;
        for(int i = 0; i < N; i++) {
            A_Pass[i][0] = pi[i] * B[i][observations[0]];
            c += A_Pass[i][0];
        }
        double probability = c;
        // Scale
        c = 1/c;
        for (int i = 0; i < N; i++) {
            A_Pass[i][0] *= c;
        }

        // Computers a-pass for the rest of the steps
        for (int t = 1; t < T; t++) {
            c = 0;
            for (int i = 0; i < N; i++) {
                A_Pass[i][t] = 0;
                for (int j = 0; j < N; j++) {
                    A_Pass[i][t] += A_Pass[j][t - 1] * A[j][i];
                }
                A_Pass[i][t] *= B[i][observations[t]];
                c += A_Pass[i][t];
            }
            probability *= c;
            //Scale
            c = 1/c;
            for (int i = 0; i < N; i++) {
                A_Pass[i][t] *= c;
            }
        }
        // Uses logarithms to prevent underflow, and to increase runtime by
        // substituting multiplication for addition.
        return probability;
    }

    /**
     * EXPERIMENTAL
     * Computes the probability of a sequence of observations.
     * Not as effecient as the other method above. No need to use.
     * @param hmm
     * @param observations
     * @return
     */
    public static double forwardBackward(HMModel hmm, int[] observations) {
        double[][] Alpha = computeAlpha(hmm, observations);
        double[][] Beta = computeBeta(hmm, observations);

        double probability = 0;
        for (int i = 0; i < Alpha.length; i++) {
            probability += Alpha[i][0] * Beta[i][0];
        }
        return probability;
    }

    /**
     * Computes a single marginal at a specific point.
     * Computes the probability of the state occuring at that time.
     *
     * @param hmm
     * @param observations
     * @param state
     * @param step
     * @return
     */
    public static double marginal(HMModel hmm, int[] observations, int state, int step) {
        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();
        int N = A.length;
        int T = observations.length;

        double[][] alphaBeta = new double[N][T];
        for (int i = 0; i < N; i++) {
            alphaBeta[i][0] = pi[i] * B[i][observations[0]];
            alphaBeta[i][T - 1] = 1;
        }
        for (int t = 1; t < step; t++) {
            for (int i = 0; i < N; i++) {
                // Computes alpha(i) based upon the sum from all arc lengths from the previous line of alphas.
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += A[j][i] * alphaBeta[j][t - 1];
                }
                sum *= B[i][observations[i]];
                alphaBeta[i][t] = sum;
            }
        }
        for (int t = T - 2; t > step; t--) {
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += A[i][j] * alphaBeta[j][t + 1];
                }
                sum *= B[i][observations[i]];
                alphaBeta[i][t] = sum;
            }
        }

        double alpha = 0;
        double beta = 0;
        for (int j = 0; j < N; j++) {
            alpha += A[j][step] * alphaBeta[j][step - 1];
            beta += B[step][j] * alphaBeta[j][step + 1];
        }
        return alpha * beta * B[step][observations[step]];
    }

    public static double[][] computeAlpha(HMModel hmm, int[] observations) {
        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();

        int N = A.length;
        int T = observations.length;

        double[][] Alpha = new double[N][T];

        // Initialize ends of alpha and beta.
        for (int i = 0; i < N; i++) {
            Alpha[i][0] = pi[i] * B[i][observations[0]];
        }
        // Inductively computes the other alphas.
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < N; i++) {
                // Computes alpha(i) based upon the sum from all arc lengths from the previous line of alphas.
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += A[j][i] * Alpha[j][t - 1] * B[i][observations[t]];
                }
                Alpha[i][t] = sum;
            }
        }
        return Alpha;
    }

    public static double[][] computeBeta(HMModel hmm, int[] observations) {
        double[][] A = hmm.getA();
        double[][] B = hmm.getB();

        int N = A.length;
        int T = observations.length;
        double[][] Beta = new double[N][T];

        for (int i = 0; i < N; i++) {
            Beta[i][T - 1] = 1;
        }
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    sum += A[i][j] * Beta[j][t + 1] * B[j][observations[t + 1]];
                }
                Beta[i][t] = sum;
            }
        }
        return Beta;
    }

    /**
     * Similar to Viterbi, ForwardBackwards computes the probability of a set
     * of observations occuring given an HmmModel.
     *
     * @param hmm
     * @param observations
     * @return probability of observations occuring
     */
    public static int[] PosteriorDecoding(HMModel hmm, int[] observations) {
        int N = hmm.getA().length;
        int T = observations.length;

        double[][] Alpha = computeAlpha(hmm, observations);
        double[][] Beta = computeBeta(hmm, observations);

        int[] states = new int[T];
        // For each step, chooses the state which maximizes
        // the probability of the observation occuring
        for(int t = 0; t < T; t++) {
            double max = -1;
            int location = -1;
            for (int i = 0; i < N; i++) {
                if (Alpha[i][t] * Beta[i][t] > max) {
                    max = Alpha[i][t] * Beta[i][t];
                    location = i;
                }
            }
            states[t] = location;
        }
        return states;
    }


    /*
     * The viterbi alogithm determines the most likely set of states given
     * a list of observations and a model.
     *
     *
     * @param A - State Transition Matrix (N x N)
     * @param B - Emmision Probability Matrix (N x M)
     * @param pi - Intial Distribution Matrix (1 x N)
     * @param observations - list of observations
     * @return the most likely set of states
     */
    public static int[] Viterbi(HMModel hmm, int[] observations) {

        double[][] A = hmm.getTransformedA();
        double[][] B = hmm.getTransformedB();
        double[] pi = hmm.getTransformedPi();

        int N = A.length;
        int T = observations.length;

        double[][] delta = new double[N][T];
        int[][] backtrance = new int[N][T];

        for (int i = 0; i < N; i++) {
            delta[i][0] = pi[i] + B[i][observations[0]];
        }
        for (int t = 1; t < T; t++) {
            // Chooses the most probable path from the last step to the current state.
            for (int i = 0; i < N; i++) {
                double max = -Double.MAX_VALUE;
                int location = -1;

                for (int j = 0; j < N; j++) {
                    double nextStep = delta[j][t - 1] + A[j][i] + B[i][observations[t]];
                    if (nextStep > max) {
                        max = nextStep;
                        location = j;
                    }
                }
                delta[i][t] = max;
                backtrance[i][t] = location;
            }
        }

        // Chooses best final state
        int[] states = new int[T];
        try {
            int bestFinalState = -1;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < N; i++) {
                double state = delta[i][T - 1];
                if (state > max) {
                    max = state;
                    bestFinalState = i;
                }
            }
            states[T - 1] = bestFinalState;

        } catch (Exception e) {
            System.out.print("Danger of underflow");
        }


        /*for (int i = 0; i < N; i++) {
            for (int j = 0; j < T; j++) {
                System.out.print(backtrance[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("endState : " + bestFinalState);
        */

        // Works backwards, choosing best state
        for (int t = T - 2; t >= 0; t--) {
            states[t] = backtrance[states[t + 1]][t + 1];
        }
        //System.out.println("Probability from Viterbi :" + Math.pow(10, delta[bestFinalState][T - 1]));
        return states;
    }

    /**
     *  Given sequences of states and observations, a model is constructed to explain it.
     *  Assumes that there are no states or observations given bigger than the largest state
     *  and observation number mentioned at least once in the data.
     * @param states
     * @param observations
     * @return Estimated Model
     */
    public static HMModel ParameterEstimation(int[][] states, int[][] observations) {
        if (!Helper.dataCorresponds(states, observations)) throw new IllegalArgumentException();
        double smoothingConstant = .01;

        int maximumState = -1;
        for (int[] test : states) {
            for (int currentState : test) {
                if (currentState > maximumState) maximumState = currentState;
            }
        }
        int maximumObservation = -1;
        for (int[] test : observations) {
            for (int currentObservation : test) {
                if (currentObservation > maximumObservation) maximumObservation = currentObservation;
            }
        }
        int N = maximumState + 1;
        int M = maximumObservation + 1;
        int trials = states.length;

        //System.out.println(N + " " + M);

        //Estimate pi
        double[] pi = new double[N];
        for (int i = 0; i < trials; i++) {
            pi[states[i][0]]++;
        }
        for (int i  = 0; i < N; i++) {
            pi[i] = (pi[i] + smoothingConstant) / (trials + smoothingConstant * N);
        }

        // Estimate A
        // Calculates chances of state each state going to every other
        double[][] A = new double[N][N];
        int[] rowTotals = new int[N];
        //Counts occurance of each possible state transition.
        for (int i = 0; i < trials; i++) {
            int next, current = states[i][0];
            for (int j = 0; j < states[i].length - 1; j++) {
                next =  states[i][j + 1];
                A[current][next]++;
                rowTotals[current]++;
                current = next;
            }
        }
        // Divides each number by the total amount of occurences of the first state
        // Makes the rows add to 1.0
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                A[i][j] = (A[i][j]  + smoothingConstant) / (rowTotals[i] + smoothingConstant * N);
            }
        }

        //Estimates B
        double[][] B = new double[N][M];
        rowTotals = new int[N];
        for (int i = 0; i < trials; i++) {
            for (int j = 0; j < states[i].length; j++) {
                int state = states[i][j];
                B[state][observations[i][j]]++;
                rowTotals[state]++;
            }
        }
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[i].length; j++) {
                B[i][j] = (B[i][j]  + smoothingConstant) / (rowTotals[i] + smoothingConstant * M);
            }
        }
        return new HMModel(A, B, pi);
    }

    /** Similar to Parameter Estimation, only the user has more control over the inner workings,
     * namely number of states and observations, and smoothing constants.
     *
     * @param states
     * @param observations
     * @param statesNumber
     * @param observationsNumber
     * @param smoothingConstant
     * @return
     */
    public static HMModel ParameterEstimation(int[][] states, int[][] observations, int statesNumber, int observationsNumber, double smoothingConstant) {
        if (smoothingConstant < 0) smoothingConstant = .01;
        if (!Helper.dataCorresponds(states, observations) || smoothingConstant > 1.0) throw new IllegalArgumentException();
        int N = 0, M = 0;
        if (statesNumber > 0) {
            N = statesNumber;
        } else {
            int maximumState = -1;
            for (int[] test : states) {
                for (int currentState : test) {
                    if (currentState > maximumState) maximumState = currentState;
                }
            }
            N = maximumState + 1;
        }
        if (observationsNumber > 0) {
            M = observationsNumber;
        } else {
            int maximumObservation = -1;
            for (int[] test : observations) {
                for (int currentObservation : test) {
                    if (currentObservation > maximumObservation) maximumObservation = currentObservation;
                }
            }
            M = maximumObservation + 1;
        }
        int trials = states.length;

        //System.out.println(N + " " + M);

        //Estimate pi
        double[] pi = new double[N];
        for (int i = 0; i < trials; i++) {
            pi[states[i][0]]++;
        }
        for (int i  = 0; i < N; i++) {
            pi[i] = (pi[i] + smoothingConstant) / (trials + smoothingConstant * N);
        }

        // Estimate A
        // Calculates chances of state each state going to every other
        double[][] A = new double[N][N];
        int[] rowTotals = new int[N];
        //Counts occurance of each possible state transition.
        for (int i = 0; i < trials; i++) {
            int next, current = states[i][0];
            for (int j = 0; j < states[i].length - 1; j++) {
                next =  states[i][j + 1];
                A[current][next]++;
                rowTotals[current]++;
                current = next;
            }
        }
        // Divides each number by the total amount of occurences of the first state
        // Makes the rows add to 1.0
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                A[i][j] = (A[i][j]  + smoothingConstant) / (rowTotals[i] + smoothingConstant * N);
            }
        }

        //Estimates B
        double[][] B = new double[N][M];
        rowTotals = new int[N];
        for (int i = 0; i < trials; i++) {
            for (int j = 0; j < states[i].length; j++) {
                int state = states[i][j];
                B[state][observations[i][j]]++;
                rowTotals[state]++;
            }
        }
        for (int i = 0; i < B.length; i++) {
            for (int j = 0; j < B[i].length; j++) {
                B[i][j] = (B[i][j]  + smoothingConstant) / (rowTotals[i] + smoothingConstant * M);
            }
        }
        return new HMModel(A, B, pi);
    }

    /**
     * Creates a blank model and iteratively loops over it to maximize the chances of the observations occuring.
     * @param observations
     * @param N - number of states
     * @param M - number of observations
     * @param maxIters - maximum number of times to loop.
     * @return
     */
    public static HMModel expectationMaximization(int[][] observations, int N, int M, int maxIters) {
        double[][] A = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = 1/N;
            }
        }
        double[][] B = new double[N][M];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                B[i][j] = 1/M;
            }
        }
        double[] pi = new double[N];
        for (int i = 0; i < N; i++) {
            pi[i] = 1/N;
        }
        return expectationMaximization(new HMModel(A, B, pi), observations, maxIters);
    }

    /**
     * Given an already exisiting HMM Model, the HMModel is changed to
     * maximize the chances of the observations occuring.
     * @param hmm
     * @param observations
     * @param maxIters - maximum number of times to loop over the model.
     * @return A HMM model with a higher probability of producing the set of observations.
     */
    public static HMModel expectationMaximization(HMModel hmm, int[][] observations, int maxIters) {
        double smoothingConstant = .01;
        if (maxIters < 0) throw new IllegalArgumentException();

        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();
        int N = A.length;
        int M = B[0].length;
        int S = observations.length;

        for (int iterations = 0; iterations < maxIters; iterations++) {
            double[] rowTotalB = new double[N];
            double[] rowTotalA = new double[N];
            double[][] newA = new double[N][N];
            double[][] newB = new double[N][M];
            double[] newPi = new double[N];

            for (int s = 0; s < S; s++) {

                double[][] alpha = computeAlpha(hmm, observations[s]);
                double[][] beta = computeBeta(hmm, observations[s]);
                int T = observations[s].length;
                double[][] marginals = Helper.multiplyMatrix(alpha, beta);

                double rowProbability = calculateProbabilityOfObservations(hmm, observations[s]);
                //Re-Estimates Pi
                for (int i = 0; i < N; i++) {
                    newPi[i] += marginals[i][0]/rowProbability;
                }

                //Re-Estimates A
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            newA[i][j] += (alpha[i][t] * beta [j][t + 1] * A[i][j] * B[j][observations[s][t + 1]]) / rowProbability;
                        }
                        rowTotalA[i] += marginals[i][t]  / rowProbability;
                    }
                }

                //Re-Estimates B
                for (int t = 0; t < T; t++) {
                    for (int i = 0; i < N; i++) {
                        newB[i][observations[s][t]] += marginals[i][t] / rowProbability;
                        rowTotalB[i] += marginals[i][t] / rowProbability;
                    }
                }

            }

            // finalizes the new parameters by dividing them by the denominators.
            for (int i = 0; i < N; i++) {
                newPi[i] = (newPi[i]  + smoothingConstant ) / (S  + smoothingConstant * N );
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    newA[i][j] = (newA[i][j]  + smoothingConstant ) / (rowTotalA[i]   + N * smoothingConstant );
                }
            }
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < M; j++) {
                    newB[i][j] = (newB[i][j]  + smoothingConstant ) / (rowTotalB[i]  + M * smoothingConstant );
                }
            }

            A = newA.clone();
            B = newB.clone();
            pi = newPi.clone();
            hmm = new HMModel(A, B, pi);
        }
        return hmm;
    }

    public static HMModel expectationMaximization2(HMModel hmm, int[][] observations, int maxIters) {
        double smoothingConstant = .01;
        if (maxIters < 0) throw new IllegalArgumentException();

        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();
        int N = A.length;
        int M = B[0].length;
        int S = observations.length;

        for (int iterations = 0; iterations < maxIters; iterations++) {
            double[][] newA = new double[N][N];
            double[][] newB = new double[N][M];
            double[] newPi = new double[N];

            for (int s = 0; s < S; s++) {

                double[][] alpha = computeAlpha(hmm, observations[s]);
                double[][] beta = computeBeta(hmm, observations[s]);
                int T = observations[s].length;
                double[][][] epsilon = new double[N][N][T];
                double[] columnTotal = new double[T - 1];
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            double transition = alpha[i][t] * beta[j][t + 1] * A[i][j] * B[j][observations[s][t + 1]];
                            epsilon[i][j][t] = transition;
                            columnTotal[t] += transition;
                        }
                    }
                }
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            epsilon[i][j][t] /= columnTotal[t];
                        }
                    }
                }
                double[][] delta = new double[N][T];
                for (int i = 0; i < N; i++) {
                    for (int t = 0; t < T; t++) {
                        for (int j = 0; j < N; j++) {
                            delta[i][t] += epsilon[i][j][t];
                        }
                    }
                }
                for (int i = 0; i < N; i++) {
                    newPi[i] += delta[i][0];
                }
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        double numerator = 0;
                        double denominator = 0;
                        for (int t = 0; t < T - 1; t++) {
                            numerator += epsilon[i][j][t];
                            denominator += delta[i][t];
                        }
                        newA[i][j] += numerator /denominator;
                    }
                }

                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < M; j++) {
                        double numerator = 0;
                        double denominator = 0;
                        for (int t = 0; t < T - 1; t++) {
                            if (observations[s][t] == j) {
                                numerator += delta[i][t];
                            }
                            denominator += delta[i][t];
                        }
                        newB[i][j] += numerator / denominator;
                    }
                }
            }
            for (int i = 0; i < N; i++) {
                newPi[i] = (newPi[i] + smoothingConstant) / (S + smoothingConstant * N);
                for (int j = 0; j < N; j++) {
                    newA[i][j] = (newA[i][j] + smoothingConstant) / (S +  smoothingConstant * N);
                }
                for (int j  = 0; j < M; j++) {
                    newB[i][j] = (newB[i][j] + smoothingConstant) / (S +  smoothingConstant * M);
                }
            }
            /*
            Helper.divideMatrix(newA, S);
            Helper.divideMatrix(newB, S);
            */
            A = newA.clone();
            B = newB.clone();
            pi = newPi.clone();
            hmm = new HMModel(A, B, pi);
        }
        return hmm;
    }

    public static HMModel expectationMaximization2withLogs(HMModel hmm, int[][] observations, int maxIters) {
        double smoothingConstant = .01;
        if (maxIters < 0) throw new IllegalArgumentException();

        double[][] A = hmm.getA();
        double[][] B = hmm.getB();
        double[] pi = hmm.getPi();

        double[][] transA = hmm.getTransformedA();
        double[][] transB = hmm.getTransformedB();
        double[] transPi = hmm.getTransformedPi();

        int N = A.length;
        int M = B[0].length;
        int S = observations.length;


        for (int iterations = 0; iterations < maxIters; iterations++) {
            double[][] newA = new double[N][N];
            double[][] newB = new double[N][M];
            double[] newPi = new double[N];
            for (int s = 0; s < S; s++) {

                double[][] alpha = computeAlpha(hmm, observations[s]);
                double[][] beta = computeBeta(hmm, observations[s]);

                double[][] logAlpha = computeAlpha(hmm, observations[s]);
                double[][] logBeta = computeBeta(hmm, observations[s]);

                for(int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        logAlpha[i][j] = Math.log(alpha[i][j]);
                    }
                    for (int j = 0; j < M; j++) {
                        logBeta[i][j] = Math.log(beta[i][j]);
                    }
                }

                int T = observations[s].length;
                double[][][] epsilon = new double[N][N][T];
                double[] columnTotal = new double[T - 1];
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            epsilon[i][j][t] = logAlpha[i][t] + logBeta[j][t + 1] + transA[i][j] + transB[j][observations[s][t + 1]];
                            columnTotal[t] += 1000000 * alpha[i][t] * beta[j][t + 1] * A[i][j] * B[j][observations[s][t + 1]];
                        }
                    }
                }
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            epsilon[i][j][t] += 6 - Math.log(columnTotal[t]);
                        }
                    }
                }
                double[][] delta = new double[N][T];
                for (int i = 0; i < N; i++) {
                    for (int t = 0; t < T; t++) {
                        for (int j = 0; j < N; j++) {
                            delta[i][t] += epsilon[i][j][t];
                        }
                    }
                }
                for (int i = 0; i < N; i++) {
                    newPi[i] += delta[i][0];
                }
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        double numerator = 0;
                        double denominator = 0;
                        for (int t = 0; t < T - 1; t++) {
                            numerator += epsilon[i][j][t];
                            denominator += delta[i][t];
                        }
                        newA[i][j] += numerator / denominator;
                    }
                }

                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < M; j++) {
                        double numerator = 0;
                        double denominator = 0;
                        for (int t = 0; t < T - 1; t++) {
                            if (observations[s][t] == j) {
                                numerator += delta[i][t];
                            }
                            denominator += delta[i][t];
                        }
                        newB[i][j] += numerator / denominator;
                    }
                }
            }
            for (int i = 0; i < N; i++) {
                newPi[i] /= S;
            }
            Helper.divideMatrix(newA, S);
            Helper.divideMatrix(newB, S);

            A = newA.clone();
            B = newB.clone();
            pi = newPi.clone();
            hmm = new HMModel(A, B, pi);
        }
        return new HMModel(A, B, pi);
    }

}
