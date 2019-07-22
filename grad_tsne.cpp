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
#include "sptree.h"
#include "grad_tsne.h"


using namespace std;

// Perform t-SNE
void Grad_TSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
               bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter) {

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

    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
    printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    bool exact = (theta == .0) ? true : false;

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
	double momentum = .5, final_momentum = .8;
	double eta = 200.0;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* dY_num = (double*) malloc(N * no_dims * sizeof(double)); 
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || dY_num == NULL || uY == NULL || gains == NULL ) 
      { printf("Memory allocation failed!\n"); exit(1); }
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
    double* log_rho = (double*) malloc(N*sizeof(double));
    double mean_logrho = 0.0;
    double var_rho = 0.0; 
    
    double* orig_densities = (double*) malloc(N*sizeof(double));
    double* emb_densities = (double*) calloc(N, sizeof(double));
    double* self_loops = (double*) calloc(N, sizeof(double)); 
    double min_beta = 0.; 
    if(exact) {

        // Compute similarities
        printf("Exact?");
        P = (double*) malloc(N * N * sizeof(double));
        if(P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        computeGaussianPerplexity(X, N, D, P, perplexity, betas, min_beta, log_rho,
				  self_loops);

        // Symmetrize input similarities
        printf("Symmetrizing...\n");
        int nN = 0;
        for(int n = 0; n < N; n++) {
	  mean_logrho += log_rho[n]/N; 
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

	for(int n=0; n<N; n++) {
	  log_rho[n] -= mean_logrho; 
	  var_rho += log_rho[n]*log_rho[n] / (N - 1); 
	}
	for(int n=0; n<N; n++) {
	  log_rho[n] /= sqrt(var_rho); 
	} 
    }

    // Compute input similarities for approximate t-SNE
    else {

        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity,
				  (int) (3 * perplexity), betas, orig_densities);

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
    }

    fstream beta_file;

    beta_file.open("beta_file.txt", fstream::out);
    if (beta_file.is_open()) {
    for (int n=0; n<N; n++) {
      beta_file << betas[n] << "\n"; 
    }
    for (int n=0; n<N; n++) {
      beta_file << orig_densities[n] << "\n"; 
    } 
    beta_file.close();
    }

    end = clock();

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

    bool dense_switch = false;
    bool dof_on = true;
    // bool dof_on = false; 
    // TEST DEBUG!!!
    // double* log_rho = (double*) calloc(N, sizeof(double)); 
    // 
    
    start = clock();

	for(int iter = 0; iter < max_iter; iter++) {

	  if (iter == max_iter - 50) {
	  // if (iter == max_iter - 1) { 
	    dense_switch = true;
	    // late exaggeration?
	    /*
	    int nN = 0;
	    
	    if (exact) {
	      for (int n=0; n<N; n++) {
		for (int m=0; m < N; m++) {
		  // P[nN + m] /= (self_loops[n] + self_loops[m]);
		  P[nN + m] *= 12.0; 
		}
		nN += N;
	      } 
	    }
	    */
	  }
	  if (iter == 2*stop_lying_iter) {
	    dof_on = false;
	    // dense_switch = true;
	  } 
        // Compute (approximate) gradient
	  if(exact) {
	    computeExactGradient(P, Y, N, no_dims, dY, betas, min_beta, 
				 log_rho, dense_switch, dof_on);

	    if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) { 
	      computeNumericalGradient(P, Y, N, no_dims, dY_num, log_rho); 
	      int nD = 0.; 
	      double delta_grad = 0.;
	      double norm_grad = 0.; 
	      for (int n=0; n < N; n++) { 
		for (int d=0; d<no_dims; d++) { 
		  delta_grad += (dY[nD+d] - dY_num[nD+d])*(dY[nD+d] - dY_num[nD+d]); 
		  norm_grad += dY[nD+d] * dY[nD + d]; 
		}
		nD += no_dims; 
	      }
	      printf("****Gradient error: %f\n", sqrt(delta_grad / norm_grad)); 
	    }
	  }
        else computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta,
			     emb_densities, sp_count, sp_time);

        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
		zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
            if(exact) { for(int i = 0; i < N * N; i++)        P[i] /= 12.0; }
            else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= 12.0; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Print out progress
        if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P, Y, N, no_dims, log_rho);
            else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
            if(iter == 0)
                printf("Iteration %d: error is %f\n", iter + 1, C);
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC);
		
		printf("Average depth: %f\n", ((double) sp_count)/(N*50));
		printf("Power computation time: %f\n",
		       (sp_time/(sp_count/100000.))); 

            }
	    start = clock();
	    sp_time = 0.;
	    sp_count = 0; 
			
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
    free(dY_num); 
    free(uY);
    free(gains);
    free(betas);
    free(orig_densities);
    free(emb_densities); 
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    printf("Fitting performed in %4.2f seconds.\n", total_time);
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void Grad_TSNE::computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, double* emb_densities, 
			   int& total_count, double& total_time)
{

    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q, total_count, total_time, emb_densities[n]);

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Approximate the gradient numerically (sanity check)
void Grad_TSNE::computeNumericalGradient(double* P, double* Y, int N, int D, double* dC,
					 double* log_rho)
{
  // cost function
  double C; 
  for(int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc(N * N * sizeof(double));
    if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double sum_Q = .0;
    double* marg_Q = (double*) malloc(N * sizeof(double)); 
    
    double* R = (double*) malloc(N*sizeof(double)); 
    double* logR = (double*) malloc(N*sizeof(double)); 

    double mean_logR = 0.; 

    int nN = 0.; 
    for(int n = 0; n < N; n++) { 
      R[n] = 0.;
      marg_Q[n] = 0.; 
      for (int m=0; m < N; m++) { 
	Q[nN + m] = 1 / (1 + DD[nN + m]); 
	marg_Q[n] += Q[nN + m]; 
	R[n] += sqrt(DD[nN + m]) * Q[nN + m]; 
      }
      nN += N; 
      R[n] /= marg_Q[n]; 
      sum_Q += marg_Q[n]; 
      logR[n] = log(R[n]); 
      mean_logR += logR[n] / N; 
    }

    double var_logR = 10.;
    double cov = 0.; 
    for (int n=0; n < N; n++) { 
      // R[n] -= mean_logR; 
      var_logR += (logR[n]-mean_logR)*(logR[n]-mean_logR) / (N - 1);
      cov += (logR[n]-mean_logR)*log_rho[n] / (N - 1); 
    } 
    // correlation = cov/(std(R) * std(rho))
    C = cov / sqrt(var_logR); 


    // Perturb each Y
    double Cprime = 0.; 
    int nD = 0.;
    
    double delta_Y = .001;
    // This will hold the updated values of the Q row
    
    // N is the overall index, which tells us which point is being perturbed
    nN = 0;
    for (int n=0; n<N; n++) {
      
      // Note that to update R_j, we only need to change one term: D_{nj} Q_{nj}
      for (int d=0; d < D; d++) { 

	int d_flag0 = 1-d; 
	int d_flag1 = d; 

	double Yd_prime[2] = { Y[n*D] + d_flag0 * delta_Y, Y[n*D + 1] + d_flag1 * delta_Y};   
	double* new_Q_row = (double*) malloc(N*sizeof(double));
	// Update Q_{nj} and marg_Q[j] for all j

	double* new_R = (double*) malloc(N*sizeof(double)); 
	double* new_marg_Q = (double*) malloc(N*sizeof(double)); 
	double* new_logR = (double*) malloc(N*sizeof(double)); 
	double new_mean = 0.; 
	double new_var = 0.; 
	double new_cov = 0.; 

	for (int j=0; j < N; j++) { 
	  double dist = (Yd_prime[0]-Y[j*D])*(Yd_prime[0]-Y[j*D]) 
	    + (Yd_prime[1]-Y[j*D+1])*(Yd_prime[1]-Y[j*D+1]); 
	  new_Q_row[j] = 1./(1. + dist);
	  new_marg_Q[j] = marg_Q[j] - Q[nN + j] + new_Q_row[j]; 
	  // Now update the densities

	  new_R[j] = R[j] - Q[nN + j] * sqrt(DD[nN + j]) + new_Q_row[j] * sqrt(dist) / marg_Q[j]; 
	  new_logR[j] = log(new_R[j]); 
	  new_mean += new_logR[j];
	} 

	// Calculate new variance and covariance
	for (int j=0; j < N; j++) { 
	  new_var += (new_logR[j] - new_mean) * (new_logR[j] - new_mean) / (N - 1); 
	  new_cov += (new_logR[j] - new_mean) * log_rho[j]; 
	} 
	Cprime = new_cov / sqrt(new_var); 
	
	dC[nD + d] = -(Cprime - C) / delta_Y; 
	free(new_Q_row);
	free(new_R); 
	free(new_marg_Q); 
	free(new_logR); 
      }
      nN += N; 
      nD += D; 
    }
    
    free(DD); 
    free(Q); 
    free(marg_Q); 
    free(R); 
    free(logR); 
}

// Compute gradient of the t-SNE cost function (exact)
void Grad_TSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC,
				     double* betas, double min_beta,
				     double* log_rho, bool dense_switch,
				     bool dof_on) {

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

    double* marg_Q = (double*) calloc(N, sizeof(double));

    double* R = (double*) malloc(N*sizeof(double));
    double mean_logR = 0.0; 

    double* E = (double*) malloc(N*sizeof(double));  

    double* d_logR = (double*) malloc(N*N *sizeof(double));

    double Cov = 0.0;
    double Var = 10.;
    
    double sum_rho_E = 0.0; 
    double sum_rho_R = 0.0;
    double sum_R_E = 0.0; 
    
    int nN = 0;
    for(int n = 0; n < N; n++) {
      R[n] = 0.0; 
    	for(int m = 0; m < N; m++) {
            if(n != m) {
	      double nu; 
	      if (dof_on) {
		nu = 1 + log(betas[n]/min_beta) + log(betas[m]/min_beta);
	      Q[nN + m] = pow(1 + DD[nN + m]/nu, -(nu + 1)/2);
	      }
	      else {
		Q[nN + m] = 1 / (1 + DD[nN + m]);
	      }
                // sum_Q += Q[nN + m];
		marg_Q[n] += Q[nN + m];
		if (dense_switch) R[n] += sqrt(DD[nN + m]) * Q[nN + m]; 
            }
        }
        nN += N;
	if (dense_switch) R[n] /= marg_Q[n]; 
	sum_Q += marg_Q[n]; 
    }
    
    // Normalize, logarithm of the R[n]
    if (dense_switch) { 
    for(int n=0; n<N; n++) {
      R[n] = log(R[n]);
      mean_logR += R[n]/N; 
    }
    // printf("mean = %.4f\n", mean_logR); 

    for(int n=0; n<N; n++) {
      Var += (R[n] - mean_logR)*(R[n] - mean_logR)/(N-1); 
    } 
    // Computing covariance:
    for(int n=0; n<N; n++) {
      Cov += (R[n] - mean_logR)*log_rho[n]/(N-1);
    }

    // Computing the residuals
    /*
    for(int n=0; n<N; n++) {
      E[n] = ((R[n] - mean_logR)/sqrt(Var) - Cov*log_rho[n]);
      sum_rho_E += log_rho[n]*E[n];
      sum_rho_R += log_rho[n]*R[n];
      sum_R_E += R[n]*E[n]; 
    } 
    */
    
    // Computing the dR/d(d_ij) derivatives
    nN = 0;
    for (int n=0; n<N; n++) {
      for (int m=n+1; m<N; m++) {
    	d_logR[nN + m] = Q[nN + m]*Q[nN + m] / marg_Q[n] * (2*sqrt(DD[nN+m]) +
    							    (1 - DD[nN +m])/exp(R[n]));
    	d_logR[m*N + n] = Q[m*N + n]*Q[m*N + n] / marg_Q[m] * (2*sqrt(DD[nN+m]) +
							     (1 - DD[nN +m])/exp(R[m]));
      }
      nN += N; 
    }
    }
    // Perform the computation of the gradient
    nN = 0;
    int nD = 0;
	for(int n = 0; n < N; n++) {
        int mD = 0;
    	for(int m = 0; m < N; m++) {
            if(n != m) {
	      double mult;
	      if (dof_on) {
		double nu = 1 + log(betas[n]/min_beta) + log(betas[m]/min_beta);
		mult = ((nu + 1)/nu)/(1 + DD[nN + m]/nu) * (Q[nN+m]/sum_Q); 
	      }
	      else mult= (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
	      double density;
	      if (dense_switch) { 
		/*
		if (m > n) {
		  density = 2*(E[n]*d_logR[nN+m] + E[m]*d_logR[m*N+n])/sqrt(Var)
		    - 2./N * sum_rho_E*(log_rho[n]*d_logR[nN+m] + log_rho[m]*d_logR[m*N + n]); 
		} 
		else {
		  density = 2*(E[m]*d_logR[nN+m] + E[n]*d_logR[m*N+n])
		    - 2./N * sum_rho_E*(log_rho[m]*d_logR[nN+m] + log_rho[n]*d_logR[m*N + n]); 
		} 

		density /= sqrt(DD[nN + m]); 
		*/
		/*
		density = 0.
		density += 2*(E[m]*d_logR[nN+m] + E[n]*d_logR[m*N+n])/sqrt(Var);

		double term = (R[m]-mean_logR)*d_logR[nN+m]+(R[n] - mean_logR)*d_logR[m*N+n]; 
		density -= 2./(Var*sqrt(Var*N))*sum_R_E*term;
		density -= 2.*sum_rho_E/(N*sqrt(Var))*(log_rho[m]*d_logR[nN+m]
							 + log_rho[n]*d_logR[m*N+n]);
		density += 2.*sum_rho_E/(N*Var*sqrt(N*Var))*term*sum_rho_R; 
		*/
		
		/*
		density = (log_rho[m]*d_logR[nN+m] + log_rho[n]*d_logR[m*N + n]); 
		*/

		density = Var*(log_rho[m]*d_logR[nN+m] + log_rho[n]*d_logR[m*N + n])
		  - Cov*((R[m]-mean_logR)*d_logR[nN+m] + (R[n]-mean_logR)*d_logR[m*N+n]);
		density /= ((N-1)*Var*sqrt(Var));

		density /= sqrt(DD[nN + m]);
	      }
                for(int d = 0; d < D; d++) {
		  // dC[nD + d] += (Y[nD + d] - Y[mD + d]) * (mult+density);
		  if(dense_switch) dC[nD + d] += (Y[nD + d] - Y[mD + d]) * (-density);
		  else dC[nD + d] += (Y[nD + d] - Y[mD + d])*mult; 
		  // dC[nD + d] += (Y[nD + d] - Y[mD + d])*mult; 
                }

		// printf("mult/density: %f\n", mult/density); 
            }
            mD += D;
		}
        nN += N;
        nD += D;
	}
	
    // Free memory
	
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
    free(R);
    free(E);
    free(d_logR); 
    free(marg_Q); 
}


// Evaluate t-SNE cost function (exactly)
double Grad_TSNE::evaluateError(double* P, double* Y, int N, int D, double* log_rho) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc(N * N * sizeof(double));
    double* Q = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    double* R = (double*) malloc(N*sizeof(double));
    double* marg_Q = (double*) calloc(N, sizeof(double)); 
    
    double mean_logR = DBL_MIN; 
    double Cov = DBL_MIN;
    double Var = DBL_MIN; 
    // Compute Q-matrix and normalization sum
    int nN = 0;
    double sum_Q = DBL_MIN;
    for(int n = 0; n < N; n++) {
      R[n] = DBL_MIN; 
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                // sum_Q += Q[nN + m];
		marg_Q[n] += Q[nN + m];
		R[n] += sqrt(DD[nN+m])*Q[nN+m]; 
            }
            else Q[nN + m] = DBL_MIN;
        }
	sum_Q += marg_Q[n];
	R[n] /= marg_Q[n]; 
        nN += N;
	R[n] = log(R[n]);
	mean_logR += R[n]/N;

	// printf("R[%i] = %f\n", n, log_rho[n]); 
    }
    for (int n=0; n<N; n++) {
      Var += (R[n] - mean_logR)*(R[n] - mean_logR) / (N-1); 
    }
    for (int n=0; n<N; n++) {
      Cov += (R[n] - mean_logR)*log_rho[n]/(N-1);
    } 
    
    for(int i = 0; i < N * N; i++) Q[i] /= sum_Q;
    printf("cov = %f, var = %f \n", Cov, Var);
    
    // Sum t-SNE error
    double C_KL = .0;
    double C_dense = .0; 
    double C = .0;
    for(int n = 0; n < N * N; n++) {
      C_KL += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	  
	  // C += pow(R[n] - mean_logR - Cov * log_rho[n], 2);
	  // printf("mean log R = %f\n", mean_logR);
	}
    for(int n = 0; n < N; n++) {
      // C_KL += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
	  
      // C_dense += pow((R[n] - mean_logR) - Cov * log_rho[n], 2);
      C_dense += (Cov/sqrt(Var+FLT_MIN))/N; 
    }

    printf("KL Err: %f, Dense Err: %f\n", C_KL, C_dense);
    C = C_KL + C_dense; 
    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
double Grad_TSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;

    int foo1 = 0;
    double foo2 = 0.; 
    double foo3 = 0.; 
    
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q, foo1, foo2, foo3);

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
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}


// Compute input similarities with a fixed perplexity
void Grad_TSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity,
					  double* betas, double& smallest_beta, double* log_rho,
					  double* self_loops) {

  double largest_beta = DBL_MIN; 
  smallest_beta = DBL_MAX; 
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N * N * sizeof(double));
    if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	computeSquaredEuclideanDistance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
    int nN = 0;
	for(int n = 0; n < N; n++) {

		// Initialize some variables
		bool found = false;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX; 
		double beta = 1.0;
		double tol = 1e-5;
        double sum_P;
	double sum_Q=DBL_MIN; 
	log_rho[n] = 0.0; 

		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
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
		
		// printf("old p: %f\n", P[nN + n + 1]); 
		// Row normalize P
		for(int m = 0; m < N; m++) {
		  sum_Q += 1./(1 + DD[nN + m]); 
		  P[nN + m] /= sum_P;
		  log_rho[n] += sqrt(DD[nN + m]) * P[nN + m];
		  // log_rho[n] += sqrt(DD[nN + m])/(1 + DD[nN + m]); 
		}
		// log_rho[n] /= sum_Q; 
		// printf("sample p: %f\n", P[nN + n + 1]);
		betas[n] = beta;
		if (beta < smallest_beta) smallest_beta = beta;
		if (beta > largest_beta) largest_beta = beta; 
		log_rho[n] = log(log_rho[n]); 
        nN += N;
	}
	/*
	double max_ratio = log(largest_beta/smallest_beta);

	nN = 0; 
	for (int n=0; n<N; n++) {
	  double extra_term = log(betas[n]/smallest_beta) / max_ratio;
	  self_loops[n] = extra_term;
	  for (int m=0; m<N; m++) {
	    if (m != n) P[nN + m] *= (self_loops[n]); 	     
	  } 
	  nN += N; 
	}
	*/
	// Clean up memory
	free(DD); DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void Grad_TSNE::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K, double* betas, double* orig_densities) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
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
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
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
	
	orig_densities[n] = 0.; 
        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) {
	  cur_P[m] /= sum_P;
	  // Distance notion of density
	  // orig_densities[n] += cur_P[m]*distances[m+1];

	  // Kernel notion of density
	  // orig_densities[n] += 1./(1 + distances[m+1]*distances[m+1]); 
	}

	
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}


// Symmetrizes a sparse matrix
void Grad_TSNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

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
void Grad_TSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
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
void Grad_TSNE::zeroMean(double* X, int N, int D) {

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
double Grad_TSNE::randn() {
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
bool Grad_TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
    fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
    fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations
	*data = (double*) malloc(*d * *n * sizeof(double));
    if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data
    if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
	fclose(h);
	printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void Grad_TSNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {

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
