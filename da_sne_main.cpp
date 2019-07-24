#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "da_sne.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main() {

    // Define some variables
	int origN, N, D, no_dims, max_iter;
	double perplexity, theta, beta_thresh, density_weight, *data;
	bool init_Y; 
    int rand_seed = -1;
    DA_SNE* tsne = new DA_SNE();
    
    // Read the parameters and the dataset
    if(tsne->load_data(&data, &origN, &D, &no_dims, &theta, &beta_thresh, &perplexity, &rand_seed, &max_iter, &init_Y, &density_weight)) {
		// Make dummy landmarks

      printf("data entry 0: %f,%f,%f\n", data[0], data[1], data[2]);
      
    // Read the parameters and the dataset
        N = origN;
	
	double* Y = (double*) malloc(N * no_dims * sizeof(double));
	
	if (init_Y && tsne->load_Y(&Y, &N, &no_dims)) printf("Initial Y loaded!\n");
	
        int* landmarks = (int*) malloc(N * sizeof(int));
        if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
        for(int n = 0; n < N; n++) landmarks[n] = n;

	// Now fire up the SNE implementation
	
	double* costs = (double*) calloc(N, sizeof(double));
        if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	if (init_Y) { 
	  tsne->run(data, N, D, Y, no_dims, perplexity, theta, beta_thresh, rand_seed, init_Y, 
		    max_iter, 0, 0, 0, density_weight);
	}
	else { 
	  tsne->run(data, N, D, Y, no_dims, perplexity, theta, beta_thresh, rand_seed, init_Y, 
		    max_iter, 250, 250, max_iter - 250, density_weight);
	} 
	// Save the results
	tsne->save_data(Y, landmarks, costs, N, no_dims);
	
        // Clean up the memory
	free(data); data = NULL;
	free(Y); Y = NULL;
	free(costs); costs = NULL;
	free(landmarks); landmarks = NULL;
    }
    delete(tsne);
}
