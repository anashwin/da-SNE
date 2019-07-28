/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include "vptree.h"
#include "da_sptree.h"
#include "da_sne.h"
#include "density_sptree.h"
// #include "cell.h"

using namespace std;

// Perform t-SNE
void DA_SNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta,
		 double beta_thresh, int rand_seed, bool skip_random_init, int max_iter,
		 int stop_lying_iter, int mom_switch_iter, int density_iter, 
		 double density_weight) {
  // JUST FOR WEIGHTED T-SNE with initialized Nu
  // density_iter = max_iter;
  
  
    // Set random seed
    if (skip_random_init != true) {
      if(rand_seed >= 0) {
          printf("Using random seed: %d\n", rand_seed);
          srand((unsigned int) rand_seed);
      } else {
          printf("Using current time as random seed...\n");
          srand(time(NULL));
      }
    }

    // printf("%.4f, %.4f\n", Y[10], Y[11]);
    
    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
    printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);

    printf("Density weight = %f\n", density_weight);  
    bool exact = (theta == .0) ? true : false;

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
    double momentum = .5, final_momentum = .8;
    double eta = 200.0;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

    // Normalize input data (to prevent numerical problems)
    printf("Computing input similarities...\n");
    start = clock();
    zeroMean(X, N, D);
    double max_X = .0;
    for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for exact t-SNE
    double* P; unsigned int* row_P; unsigned int* col_P; double* val_P; 
    double* betas = (double*) malloc(N*sizeof(double));
    double* orig_densities = (double*) malloc(N*sizeof(double));
    double* emb_densities = (double*) calloc(N, sizeof(double));
    
    double* log_betas = (double*) malloc(N*sizeof(double)); 
    double* self_loops = (double*) malloc(N*sizeof(double)); 
    
    double min_beta;
    double max_beta; 
    if(exact) {
        // Compute similarities
        printf("Exact?");
        P = (double*) malloc(N * N * sizeof(double));
        if(P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeGaussianPerplexity(X, N, D, P, perplexity, betas, min_beta, max_beta);

        // Symmetrize input similarities
        printf("Symmetrizing...\n");
        int nN = 0;
        for(int n = 0; n < N; n++) {
            int mN = (n + 1) * N;
            for(int m = n + 1; m < N; m++) {
                P[nN + m] += P[mN + n];
                P[mN + n]  = P[nN + m];
                mN += N;
            }
            nN += N;
        }
        double sum_P = .0;
        for(int i = 0; i < N * N; i++) sum_P += P[i];
        for(int i = 0; i < N * N; i++) P[i] /= sum_P;
	// cout << "min beta: " << min_beta; 
    }

    // Compute input similarities for approximate t-SNE
    else {

        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity,
				  (int) (3 * perplexity),
				  betas, min_beta, max_beta, self_loops, orig_densities);

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
	// Should the downweighting happen here or in the computeGaussianPerplexity function?
    }

    // save_betas(betas);
    fstream beta_file;

    beta_file.open("beta_file.txt", fstream::out);
    if (beta_file.is_open()) {
    for (int n=0; n<N; n++) {
      log_betas[n] = log(betas[n]/min_beta);
      // log_betas[n] = log(max_beta/betas[n]); 
      beta_file << betas[n] << "\n"; 
    }
    for (int n=0; n<N; n++) {
      beta_file << orig_densities[n] << "\n";
    } 
    
    beta_file.close();
    }
    end = clock();
    
    double* log_orig_densities = (double*) malloc(N*sizeof(double)); 
    double mean_od = 0.; 
    double var_od = 0.; 
    for (int n=0; n<N; n++) {
      log_orig_densities[n] = log(orig_densities[n]); 
      mean_od += log_orig_densities[n]/N; 
    }
    for (int n=0; n<N; n++) { 
      var_od += (log_orig_densities[n] - mean_od)*(log_orig_densities[n] - mean_od) / (N-1); 
    } 
    for (int n=0; n<N; n++) { 
      log_orig_densities[n] -= mean_od; 
      log_orig_densities[n] /= sqrt(var_od);
    } 

    // Lie about the P-values
    
    if(exact) { for(int i = 0; i < N * N; i++)        P[i] *= 12.0; }
    else {      for(int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0; }
    
	// Initialize solution (randomly)
  if (skip_random_init != true) {
  	for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
  }

	// Perform main training loop
    if(exact) printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    else printf("Input similarities computed in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));


    int sp_count = 0; 
    double sp_time = 0.; 
    double avg_time = 0.; 
    
    start = clock();

    for(int iter = 0; iter < max_iter; iter++) {
	  // printf("Y = %.4f, %.4f\n", Y[10], Y[11]);
	  
        // Compute (approximate) gradient
	  if(exact) computeExactGradient(P, Y, N, no_dims, dY, betas, min_beta, max_beta);
	  else computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, log_betas,
			       emb_densities, log_orig_densities, 
			       min_beta, max_beta, beta_thresh, D, self_loops, sp_count, sp_time,
			       iter < stop_lying_iter, iter >= density_iter, 
			       density_weight);

	  /*
	  for (int i=1500; i<1550; i++) {
	    printf("Beta = %f\n AND min_beta = %f\n", log_betas[i], log(min_beta)); 
	    for (int j=100; j<110; j++) {
	      printf("Nu = %f\n", 1 + (log_betas[i] + log_betas[j]));
	    } 
	  } 
	  */

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .1) : (gains[i] * .3);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
	for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
	zeroMean(Y, N, no_dims);
	
	// DEBUG!
	/*
	for (int n=0; n < N; n++) { 
	  printf("( %f, %f )\n", Y[2*n], Y[2*n+1]); 
	} 
	*/

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
            else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
	    /*
	    else {
	      for(int row=0; row<N; row++) {
		for(int j=row_P[row]; j < row_P[row+1]; j++) {
		  // val_P[j] /= (12.0 * (self_loops[row] + .0001));
		  val_P[j] *= (self_loops[row] + .0001)/12.0; 
		} 
	      } 
	    } 
	    /*
	    else      {
	      int row_ind = 0; 
	      for(int i = 0; i < row_P[N]; i++) {
		if (row_P[row_ind + 1] == i) row_ind++; 
		val_P[i] /= (12.0 * self_loops[row_ind]);
	      }
	    }
	    */
	    
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if (iter >= 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P, Y, N, no_dims);
            else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta, betas, min_beta,
					beta_thresh);  // doing approximate computation here!
            if(iter == 0)
                printf("Iteration %d: error is %f\n", iter + 1, C);
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);

		printf("Average depth: %f\n", ((double) sp_count)/(50*N));
		printf("Power computation time: %f\n",
		       sp_time/(((double) sp_count)/100000.)); 
		
            }
	    sp_count = 0;
	    sp_time = 0.; 
	    
	    start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    beta_file.open("beta_file.txt", fstream::out | fstream::app);
    if (beta_file.is_open()) {
      for (int n = 0; n<N; n++) {
	beta_file << emb_densities[n] << "\n";
	// printf("%f\n", emb_densities[n]); 
      } 
    } 
    
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    free(betas);
    free(orig_densities);
    free(emb_densities); 
    free(log_betas); 
    free(self_loops); 
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    printf("Fitting performed in %4.2f seconds.\n", total_time);
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void DA_SNE::computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, 
			     double* Y, int N, int D, double* dC, double theta,
			     double* betas, double* emb_densities, 
			     double* log_orig_densities, 
			     double beta_min, double beta_max, double beta_thresh, int orig_D,
			     double* self_loops, int& total_count, double& total_time,
			     bool lying, bool density, double density_weight)
{

  
  // DEBUG!!
  //  density = false; 
  // for(int n=0; n<N; n++) {
  //   printf("beta: %4.2f\n", betas[n]); 
  // } 
    // Construct space-partitioning tree on current map
  DA_SPTree* tree = new DA_SPTree(D, Y, betas, beta_min, N);
  SPTree* d_tree; 
    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;

    /*
    for (int n=0; n<N; n++) {
      // double beta = betas[n];
      // sum_Q += .5*((beta_max/beta) - 1 + log(beta/beta_max))/(N);
      // sum_Q += exp(-beta/beta_min);
      sum_Q = .0;
      // sum_Q += self_loops[n]/log(N);
    } 
    */
    
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    double* dense_f1 = (double*) calloc(N * D, sizeof(double));
    double* dense_f2 = (double*) calloc(N * D, sizeof(double)); 

    double mean_ed = 0.; 
    double var_ed = 10.; 
    double cov_ed = 0.; 

    double marg_Q = 0.;


    // double emb_density = 0.;
    lying = false;
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, lying);
    if (lying) {
      
      for(int n = 0; n < N; n++) {
	double aux_sum = 0.; 
	tree->computeNonEdgeForces(n, theta, beta_thresh, neg_f + n * D,
				   &marg_Q, total_count, total_time, emb_densities[n], 
				   aux_sum);
	sum_Q += marg_Q;
	// emb_densities[n] /= marg_Q;
	emb_densities[n] = (emb_densities[n] + log(marg_Q) * aux_sum) / marg_Q; 

	// Marginal notion of density 
	// emb_densities[n] = marg_Q; 
	marg_Q = 0.;
      }
    }
    else if(density) { 

      double* all_marg_Q = (double*) malloc(N * sizeof(double)); 
      double* log_emb_densities = (double*) malloc(N * sizeof(double)); 
      double* emb_densities_no_entropy = (double*) malloc(N*sizeof(double)); 
      for(int n = 0; n < N; n++) {
	double aux_sum = 0.; 
	tree->computeNonEdgeForces(n, theta, neg_f + n * D,
				   &marg_Q, total_count, total_time, emb_densities[n], aux_sum);
	
	// emb_densities[n] /= marg_Q;
	emb_densities[n] = (emb_densities[n] + log(marg_Q) * aux_sum) / marg_Q;
	emb_densities_no_entropy[n] = aux_sum / marg_Q; 
	all_marg_Q[n] = marg_Q; 
	sum_Q += marg_Q;
	marg_Q = 0.;	
      }
      
      // Compute mean of the log embedded densities
      for(int n=0; n < N; n++) { 
	log_emb_densities[n] = log(emb_densities[n]); 
	mean_ed += log_emb_densities[n]/N; 
      }

      for(int n=0; n < N; n++) {
	log_emb_densities[n] -= mean_ed; // Center log embedded densities
	var_ed += log_emb_densities[n]*log_emb_densities[n] / (N - 1);
	cov_ed += log_emb_densities[n]*log_orig_densities[n] / (N - 1);
      } 

      // DEBUG!
      // printf("Creating the Density SPTree! with var %f and covar %f \n", var_ed, cov_ed); 
      d_tree = new SPTree(D, Y, emb_densities, log_emb_densities,
			  emb_densities_no_entropy, log_orig_densities, 
			  all_marg_Q, N); 


      for(int n=0; n < N; n++) { 
	d_tree -> computeDensityForces(n, theta, dense_f1 + n*D, dense_f2 + n*D); 
      }

      free(emb_densities_no_entropy); 
      free(log_emb_densities); 
      free(all_marg_Q); 
      delete d_tree; 
    }  
    else {
      for(int n = 0; n < N; n++) {
	double trash = 0.; 
	tree->computeNonEdgeForces(n, theta, neg_f + n * D,
				   &marg_Q, total_count, total_time, emb_densities[n], trash); 
	
	emb_densities[n] /= marg_Q; 
	sum_Q += marg_Q;
	marg_Q = 0.;	
      }
    }
    
    double std_ed = sqrt(var_ed) + DBL_MIN; // standard deviation
    double std_var_ed = std_ed * var_ed + DBL_MIN; // variance ^ 3/2

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
      // dC[i] = betas[(int)(i/D)]*pos_f[i] - (neg_f[i] / sum_Q);
      dC[i] = pos_f[i] - (neg_f[i] / sum_Q); 
      if (density) { 
	dC[i]-=density_weight*(dense_f1[i]/std_ed-cov_ed*dense_f2[i]/std_var_ed)/(N-1); 
      } 
    }
    /*
    printf("dC: %f; pos_f: %f; neg_f: %f; sum_Q: %f\n",dC[0], pos_f[0], neg_f[0],
	   sum_Q); 
    */
    free(pos_f);
    free(neg_f);
    free(dense_f1);
    free(dense_f2); 
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
void DA_SNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC, double* betas, double beta_min, double beta_max) {

	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0;

    // Here we take into account the diagonals?
    for(int n=0; n<N; n++) {
      sum_Q += 1.5*(beta_min/betas[n] - 1 + log(betas[n]/beta_min));
      // sum_Q += exp(-betas[n]/beta_max);
      sum_Q = .0; 
     } 
    
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {

            if(n != m) {

	      // double nu = 1+log(betas[n]) + log(betas[m]) - 2*log(beta_min);
	      double nu = (betas[n] + betas[m])/(2.*beta_min);
	      
	      // double t_scale = betas[m]*betas[n]/(beta_min*(betas[m] + betas[n])); 
	      // printf("nu = %f\n", nu);
	      // nu = 1; 
	      // Q[nN + m] = pow(1 + DD[nN + m]/nu,-(nu+1)/2.);
	      Q[nN + m] = 1.0/(1 + nu*DD[nN+m]); 
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	// Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {

	      // double nu = 1+log(betas[n]) + log(betas[m]) - 2*log(beta_min);
	      double nu = (betas[n] + betas[m])/(2.*beta_min); 

	      // double t_scale = betas[m]*betas[n]/(beta_min*(betas[m] + betas[n])); 
	      // nu = 1; 
	      double mult = nu*(P[nN + m] - (Q[nN + m] / sum_Q))/(1+DD[nN+m]*nu);
                for(int d = 0; d < D; d++) {
                    dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
                }
            }
            mD += D;
		}
        nN += N;
        nD += D;
	}

    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (exactly)
double DA_SNE::evaluateError(double* P, double* Y, int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
            else Q[nN + m] = DBL_MIN;
        }
        nN += N;
    }
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;

    // Sum t-SNE error
    double C = .0;
	for(int n = 0; n < N * N; n++) {
        C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	}

    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
double DA_SNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* betas, double beta_min, double beta_thresh)
{

    // Get estimate of normalization term
  DA_SPTree* tree = new DA_SPTree(D, Y, betas, beta_min, N);


  int foo1;
  double foo2; 
  double foo3 = 0.; 
  double foo4 = 0.; 
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, beta_thresh, buff, &sum_Q,
							  foo1, foo2, foo3, foo4);
    
    printf("sum: %f\n", sum_Q); 
    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
	    // double nu = (int) (log(betas[n]) + log(betas[col_P[i]]) - 2*log(beta_min));
	    double nu = 1.;
            Q = pow(1.0 + Q/nu, -(nu+1)/2.0) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Compute input similarities with a fixed perplexity
void DA_SNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity, double* betas, double& smallest_beta, double& largest_beta) {
  largest_beta = DBL_MIN; 
  smallest_beta = DBL_MAX;
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);


	// Comparing our Euclidean distances:
	int tempnN = 0;
	double* avg_dist = (double*) malloc(N*sizeof(double)); 
	for (int n=0; n<N; n++) {
	  avg_dist[n] = 0.; 
	  for (int m=0; m<N; m++) {
	    avg_dist[n] += DD[tempnN+m];  
	  }
	  avg_dist[n] /= N;
	  // printf("avg of n=%d: %f\n", n, avg_dist[n]);
	  tempnN += N; 
	} 

	printf("0,300: %f\n", DD[300]);
	printf("X1: %f, %f, %f; X2: %f, %f, %f\n", X[0], X[1], X[2], X[900], X[901], X[902]); 
	
	double* sums_P = (double*)malloc(N*sizeof(double));
	// double* sums_Q = (double*)malloc(N*sizeof(double));
	
	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
		double sum_P;

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
		  double min_D = DBL_MAX;
		  for(int m = 0; m < N; m++) {
		    P[nN + m] = exp(-beta * DD[nN + m]);
		    if (DD[nN+m] < min_D) min_D = DD[nN+m]; 
		  }
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}

		sums_P[n] = sum_P; 
		// Row normalize P

		// We can add something like 1./beta 
		// for(int m = 0; m < N; m++) P[nN + m] /= (sqrt(2.*3.14159/beta)+sum_P);
		/*
		for(int m = 0; m < N; m++) {
		  P[nN + m] /= (sum_P);
		}
		*/

        nN += N;
	betas[n] = beta;
	// printf("beta: %f\n", beta);
	if (beta < smallest_beta) smallest_beta = beta;
	if (beta > largest_beta) largest_beta = beta; 
	}

	nN = 0;
	for (int n = 0; n < N; n++) {
	  for (int m =0; m < N; m++) {

	    // P[nN+m] /= .5*D*betas[n]/largest_beta + sums_P[n]; 
	    
	    // KL-divergence b/w Gaussians downweighting
	    
	    // P[nN+m] /= (.5*D*(log(largest_beta/betas[n])
	    // - 1 + betas[n]/largest_beta)+sums_P[n]);

	    // squared exponential downweighting
	    // P[nN+m] /= exp(-betas[n]/largest_beta)+sums_P[n]; 
	    
	    
	    P[nN+m] /= (.5*D*(log(betas[n]/largest_beta)
			      - 1 + largest_beta/betas[n]) + sums_P[n]); 
	    
	    //P[nN+m] /= sums_P[n];
	  }
	    nN += N;

	    // printf("correction: %f\n", .5*D*(log(largest_beta/betas[n]) - 1 + betas[n]/largest_beta)); 
	  }

	
	printf("min beta: %f\n", smallest_beta);
	printf("max beta: %f\n", largest_beta); 
	// Clean up memory
	free(DD); DD = NULL;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void DA_SNE::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P,
				       unsigned int** _col_P, double** _val_P, double perplexity,
				       int K, double* betas, double& smallest_beta,
				       double& largest_beta, double* self_loops,
				       double* orig_density)
{
    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    largest_beta = DBL_MIN;
    smallest_beta = DBL_MAX; 
    
    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));

    double* sums_P = (double*) malloc(N*sizeof(double)); 
    double* sums_Q = (double*) malloc(N*sizeof(double));
    
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
	    // In this loop, we also compute our (unnormalized) density measure
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) {
	      sum_P += cur_P[m];
	    }
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

	betas[n] = beta;
	if (beta < smallest_beta) smallest_beta = beta;
	if (beta > largest_beta) largest_beta = beta; 
	
        // Row-normalize current row of P and store in matrix
	/*
        for(unsigned int m = 0; m < K; m++) {
	  cur_P[m] /= sum_P; // incorporate P downweighting
	}
	*/

	sums_P[n] = sum_P;

	double aux_sum = 0.; 
	
	orig_density[n] = 0.;
	sums_Q[n] = 0.; 
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
	    // orig_density[n] += cur_P[m]*distances[m+1]/sum_P;
	    orig_density[n] += (distances[m+1]/(1 + distances[m+1]*distances[m+1])
				* log(1 + distances[m+1]*distances[m+1]));

	    aux_sum += distances[m+1] / (1 + distances[m+1]*distances[m+1]);
	
	    sums_Q[n] += 1./(1. + distances[m+1]*distances[m+1]); 
        }
	// orig_density[n] /= sums_Q[n];
	orig_density[n] = (orig_density[n] + log(sums_Q[n]) * aux_sum) / sums_Q[n]; 
	
	/*
	orig_density[n] = sqrt(beta)*(1 + sum_P);
	// orig_density[n] = 0.; 
	// Marginal notion of density 
	for(unsigned int m = 0; m < K; m++) {
	  col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
	  val_P[row_P[n] + m] = cur_P[m];
	  // orig_density[n] += 1./(1 + distances[m+1]*distances[m+1]); 
	} 
	*/
    }

    double max_ratio = log(largest_beta/smallest_beta); 
    for(unsigned int n=0; n<N; n++) {

      // double extra_term = (.5*((largest_beta/betas[n]) - 1 + log(betas[n]/largest_beta))/(log(betas[n]/smallest_beta)*D));
      // double extra_term = log(betas[n]/smallest_beta)/N;
      // double extra_term = (.5*((largest_beta/betas[n]) - 1 + log(betas[n]/largest_beta))/(D));
      // double extra_term = (betas[n] - smallest_beta) / (largest_beta - smallest_beta);
      double extra_term = log(betas[n]/smallest_beta) / max_ratio;
      
      for(unsigned int m=0; m<K; m++) {

	// self_loops[n] = (extra_term < 10*sums_P[n]) ? extra_term : 10*sums_P[n];
	// self_loops[n] = (extra_term < 12.0) ? extra_term : 12.0; 
	// self_loops[n] = extra_term;
	self_loops[n] = 0.;

	// val_P[row_P[n] + m] *= 12.0/(self_loops[n] + sums_P[n]); 
	val_P[row_P[n] + m] *= (1.+self_loops[n])/ sums_P[n];
	
	// val_P[row_P[n] + m] *= 12.0/ sums_P[n]; 
	// val_P[row_P[n] + m] /=  (self_loops[n] + sums_P[n]);
	// self_loops[n] = sums_P[n] / (self_loops[n] + sums_P[n]); 
	// val_P[row_P[n] + m] *= (12.0 + self_loops[n])/sums_P[n];
	// val_P[row_P[n] + m] *= 12.0 + self_loops[n]; 
	// val_P[row_P[n]+m] /= (exp(-betas[n]/smallest_beta)+sums_P[n]);
	
      }
      // printf("sum = %.4f, out of %.4f\n", self_loops[n], sums_P[n]); 
    } 
    
    // Clean up memory
    obj_X.clear();
    free(cur_P);
    free(sums_P); 
    delete tree;
}


// Symmetrizes a sparse matrix
void DA_SNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
void DA_SNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Makes data zero-mean
void DA_SNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
double DA_SNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool DA_SNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* beta_thresh, double* perplexity, int* rand_seed, int* max_iter,
		       bool* init_Y, double* density_weight) {

	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
    fread(theta, sizeof(double), 1, h);										// gradient accuracy
    fread(beta_thresh, sizeof(double), 1, h); // beta threshold value
	fread(perplexity, sizeof(double), 1, h);								// perplexity
    fread(density_weight, sizeof(double),1,h); // weight of the density term
	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
    fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
    fread(init_Y, sizeof(bool),1,h); // Is there a Y to be loaded? 
    
    printf("DO SOMETHING?!?!?! %f\n", *density_weight); 

    *data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data
    
    if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
	fclose(h);
	printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

bool DA_SNE::load_Y(double** Y, int* n, int* d) {

  FILE *h;
  if ((h = fopen("Y_init.dat", "r+b")) == NULL) {
    printf("Error: could not open Y init file.\n");
    return false; 
  }
  fread(n, sizeof(int), 1, h);
  fread(d, sizeof(int), 1, h);

  *Y = (double*) malloc(*d * *n * sizeof(double));
  if(*Y == NULL) { printf("Memory allocation failed!\n"); exit(1); }

  fread(*Y, sizeof(double), *n * *d, h);
  fclose(h);
  printf("Read the %i x %i initial out matrix successfully!\n", *n, *d); 
  return true; 
  
} 

// Function that saves map to a t-SNE file
void DA_SNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen("result.dat", "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
    fwrite(costs, sizeof(double), n, h);
    fclose(h);
	printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
