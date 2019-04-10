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
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND` ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef DA_SNE_H
#define DA_SNE_H


static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class DA_SNE
{
public:
  void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, double beta_thresh, int rand_seed,
             bool skip_random_init, int max_iter=1000, int stop_lying_iter=250, int mom_switch_iter=250);
  bool load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* beta_thresh, double* perplexity, int* rand_seed, int* max_iter, bool* init_Y);
  bool load_Y(double** Y, int* n, int* d); 
    void save_data(double* data, int* landmarks, double* costs, int n, int d);
    void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N); // should be static!


private:
    void computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P,
			 double* Y, int N, int D, double* dC, double theta, double* betas, 
			 double beta_min, double beta_max, double beta_thresh, int orig_D,
			 double* self_loops, int& total_count, double& total_time, bool lying);
    void computeExactGradient(double* P, double* Y, int N, int D, double* dC, double* betas,
			      double beta_min, double beta_max);
    double evaluateError(double* P, double* Y, int N, int D);
    double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* betas, double beta_min, double beta_thresh);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity, double* betas, double& smallest_beta, double& largest_beta);
    // Need to update the nearest neighbors P calculation too 
    void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P,
				   unsigned int** _col_P, double** _val_P, double perplexity, int K,
				   double* betas, double& smallest_beta, double& largest_beta, double* self_loops);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    double randn();
};

#endif

