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

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include "cell.h"
#include "da_sptree.h"

/*
// Constructs cell
Cell::Cell(unsigned int inp_dimension) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
}

Cell::Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
    for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
    for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell
Cell::~Cell() {
    free(corner);
    free(width);
}

double Cell::getCorner(unsigned int d) {
    return corner[d];
}

double Cell::getWidth(unsigned int d) {
    return width[d];
}

void Cell::setCorner(unsigned int d, double val) {
    corner[d] = val;
}

void Cell::setWidth(unsigned int d, double val) {
    width[d] = val;
}

// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{
    for(int d = 0; d < dimension; d++) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}
*/

// Default constructor for DA_SPTree -- build tree, too!
DA_SPTree::DA_SPTree(unsigned int D, double* inp_data, double* inp_betas, double beta_min, unsigned int N)
{
    
    // Compute mean, width, and height of current map (boundaries of DA_SPTree)
    int nD = 0;
    double* mean_Y = (double*) calloc(D,  sizeof(double));
    double*  min_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  min_Y[d] =  DBL_MAX;
    double*  max_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  max_Y[d] = -DBL_MAX;
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int d = 0; d < D; d++) {
            mean_Y[d] += inp_data[n * D + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) mean_Y[d] /= (double) N;
    
    // Construct DA_SPTree
    double* width = (double*) malloc(D * sizeof(double));
    for(int d = 0; d < D; d++) width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    init(NULL, D, inp_data, inp_betas, beta_min, mean_Y, width);
    fill(N);
    
    // Clean up memory
    free(mean_Y);
    free(max_Y);
    free(min_Y);
    free(width);
}

// Constructor for DA_SPTree with particular size and parent -- build the tree, too!
DA_SPTree::DA_SPTree(unsigned int D, double* inp_data, double* inp_betas, double beta_min, unsigned int N, double* inp_corner, double* inp_width)
{
  init(NULL, D, inp_data, inp_betas, beta_min, inp_corner, inp_width);
    fill(N);
}


// Constructor for DA_SPTree with particular size (do not fill the tree)
DA_SPTree::DA_SPTree(unsigned int D, double* inp_data, double* inp_betas, double beta_min, double* inp_corner, double* inp_width)
{
  init(NULL, D, inp_data, inp_betas, beta_min, inp_corner, inp_width);
}


// Constructor for DA_SPTree with particular size and parent (do not fill tree)
DA_SPTree::DA_SPTree(DA_SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_betas, double beta_min, double* inp_corner, double* inp_width) {
  init(inp_parent, D, inp_data, inp_betas, beta_min, inp_corner, inp_width);
}


// Constructor for DA_SPTree with particular size and parent -- build the tree, too!
DA_SPTree::DA_SPTree(DA_SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_betas,
		     double beta_min, unsigned int N, double* inp_corner, double* inp_width)
{
  init(inp_parent, D, inp_data, inp_betas, beta_min, inp_corner, inp_width);
    fill(N);
}


// Main initialization function
void DA_SPTree::init(DA_SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_betas, double beta_min,
		     double* inp_corner, double* inp_width)
{
    parent = inp_parent;
    dimension = D;
    no_children = 2;
    for(unsigned int d = 1; d < D; d++) no_children *= 2;
    data = inp_data;
    
    betas = inp_betas; 
    is_leaf = true;
    size = 0;
    cum_size = 0;

    log_beta_com = 0.; 
    overall_beta_min = log(beta_min); 
    
    min_beta = DBL_MAX;
    max_beta = DBL_MIN; 

    
    boundary = new Cell(dimension);
    for(unsigned int d = 0; d < D; d++) boundary->setCorner(d, inp_corner[d]);
    for(unsigned int d = 0; d < D; d++) boundary->setWidth( d, inp_width[d]);
    
    children = (DA_SPTree**) malloc(no_children * sizeof(DA_SPTree*));
    for(unsigned int i = 0; i < no_children; i++) children[i] = NULL;

    center_of_mass = (double*) malloc(D * sizeof(double));
    for(unsigned int d = 0; d < D; d++) center_of_mass[d] = .0;
    
    buff = (double*) malloc(D * sizeof(double));
}


// Destructor for DA_SPTree
DA_SPTree::~DA_SPTree()
{
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i] != NULL) delete children[i];
    }
    free(children);
    free(center_of_mass);
    free(buff);
    delete boundary;
}


// Update the data underlying this tree
void DA_SPTree::setData(double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
DA_SPTree* DA_SPTree::getParent()
{
    return parent;
}


// Insert a point into the DA_SPTree
bool DA_SPTree::insert(unsigned int new_index)
{
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * dimension;
    if(!boundary->containsPoint(point))
        return false;
    
    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] *= mult1;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] += mult2 * point[d];

    double beta = betas[new_index];
    
    // update beta values
    log_beta_com = ((double) (cum_size - 1))/ ((double) cum_size) * log_beta_com
      + 1.0/((double) cum_size) * beta; // New center of mass

    if (beta < min_beta) min_beta = beta;
    if (beta > max_beta) max_beta = beta;

    beta_ratio = max_beta/min_beta - 1; // i.e. beta_max = (1+beta_ratio)*beta_min
    
    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }
    
    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned int n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned int d = 0; d < dimension; d++) {
            if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) return true;
    
    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();
    
    // Find out where the point can be inserted
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i]->insert(new_index)) return true;
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}

    
// Create four children which fully divide this cell into four quads of equal area
void DA_SPTree::subdivide() {
    
    // Create new children
    double* new_corner = (double*) malloc(dimension * sizeof(double));
    double* new_width  = (double*) malloc(dimension * sizeof(double));
    for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        for(unsigned int d = 0; d < dimension; d++) {
            new_width[d] = .5 * boundary->getWidth(d);
            if((i / div) % 2 == 1) new_corner[d] = boundary->getCorner(d) - .5 * boundary->getWidth(d);
            else                   new_corner[d] = boundary->getCorner(d) + .5 * boundary->getWidth(d);
            div *= 2;
        }
        children[i] = new DA_SPTree(this, dimension, data, betas, overall_beta_min, new_corner, new_width);
    }
    free(new_corner);
    free(new_width);
    
    // Move existing points to correct children
    for(unsigned int i = 0; i < size; i++) {
        bool success = false;
        for(unsigned int j = 0; j < no_children; j++) {
            if(!success) success = children[j]->insert(index[i]);
        }
        index[i] = -1;
    }
    
    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build DA_SPTree on dataset
void DA_SPTree::fill(unsigned int N)
{
    for(unsigned int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
bool DA_SPTree::isCorrect()
{
    for(unsigned int n = 0; n < size; n++) {
        double* point = data + index[n] * dimension;
        if(!boundary->containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(int i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in DA_SPTree
void DA_SPTree::getAllIndices(unsigned int* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in DA_SPTree
unsigned int DA_SPTree::getAllIndices(unsigned int* indices, unsigned int loc)
{
    
    // Gather indices in current quadrant
    for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;
    
    // Gather indices in children
    if(!is_leaf) {
        for(int i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
    }
    return loc;
}


unsigned int DA_SPTree::getDepth() {
    if(is_leaf) return 1;
    int depth = 0;
    for(unsigned int i = 0; i < no_children; i++) depth = fmax(depth, children[i]->getDepth());
    return 1 + depth;
}
/*
// Compute the local densities
void DA_SPTree::computeDensityForces(unsigned int point_index, double theta, double neg_f[],
				     double* sum_Q, int& total_count, double& total_time,
				     double& emb_density) { 
// Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;
    
    // Compute distance between point and center-of-mass
    double D = .0;
    unsigned int ind = point_index * dimension;
    for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
    
    // Check whether we can use this node as a "summary"
    double max_width = 0.0;
    double cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
        cur_width = boundary->getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }

    if(is_leaf || max_width / sqrt(D) < theta) {
      double dist = sqrt(D);
      D = 1.0 / (1.0 + D);
      *sum_Q += mult; 
    } 

} 
*/

// Compute non-edge forces using Barnes-Hut algorithm (with the full DA_SNE algorithm)
void DA_SPTree::computeNonEdgeForces(unsigned int point_index, double theta, double beta_thresh, double neg_f[], double* sum_Q, int& total_count, double& total_time, double& emb_density)
{
  double tol = 1e-5; 
    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;
    
    // Compute distance between point and center-of-mass
    double D = .0;
    unsigned int ind = point_index * dimension;
    for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
    
    // Check whether we can use this node as a "summary"
    double max_width = 0.0;
    double cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
        cur_width = boundary->getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }

    // Check the beta condition as well:
    // if (!is_leaf) printf("beta ratio: %f\n", beta_ratio);

    // if (max_width / sqrt(D) < theta) printf("ind: %i, thresh: %f\n", point_index, beta_ratio);
    
    // if(is_leaf || (max_width / sqrt(D) < theta && beta_ratio < beta_thresh)) {
    double beta = betas[point_index]; 

    // LOGISTIC FUNCTION?
    // -L/(1 + exp(-(beta/overall_beta_min-1)));

    // -2*gap/(1 + exp(-(beta/overall_beta_min - 1))) + gap + init; 

    double gap = 0.1; 
    // double condition = -2*gap/(1 + exp(-.25*(beta))) + 2*gap + theta;
    // if (is_leaf || (max_width / sqrt(D) < condition)) {
    // if (is_leaf || (max_width / sqrt(D) < (1 + exp(overall_beta_min - beta))*theta)) { 
      // if (is_leaf || (max_width / sqrt(D) < (1 + overall_beta_min/beta) * theta)) {
      // if (is_leaf || (max_width / sqrt(D) < (1 + exp(-beta))*theta)) { 
    if (is_leaf || (max_width / sqrt(D) < theta)) {
        // Compute and add t-SNE force between point and current node
      // TO DO: Updated D to take into account the degrees of freedom
      // Ideally we should handle different DoF definitions
      double dist = sqrt(D); 
      clock_t start = clock(); 
      // double nu = (1 + atan(beta + log_beta_com));
      double nu = 1 + beta + log_beta_com; 
      // double nu = 1.; 
      // double nu = (betas[point_index] + exp(log_beta_com))/(2.*overall_beta_min);
      
      double D_base = 1.0/(1.0 + D/nu);
      D = pow(1.0 + D/nu, (-(nu+1)/2.0));

      // double D_base = 1./(1. + D);
      // D = pow(1. + D, -nu/2.);
      
      // D = 1.0/(1 + nu *D); 
      
      /* 
      D = 1.0;
      for (int i=0; i<(int)((nu+1)/2.); i++) { 
	D *= D_base;
      }
      */
      
      double mult = cum_size * D;
      *sum_Q += mult; // This is going to be Z
      
      // Updating to log of expectation
      emb_density += mult*log(dist + tol);
      //
      
      // emb_density += mult; 
      mult *= (nu+1)/nu*D_base;
      // mult *= nu/2. * D_base; 
      // mult *= D_base; 
      // mult *= nu*D; 
      clock_t end = clock();
	// printf("%f %f %f %f\n", neg_f[0], neg_f[1], mult, betas[point_index]); 
      for(unsigned int d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
      total_time += (float) (end - start) / CLOCKS_PER_SEC;
      total_count ++; 
    }
    else {

        // Recursively apply Barnes-Hut to children
      for(unsigned int i = 0; i < no_children; i++) children[i]->computeNonEdgeForces(point_index, theta, beta_thresh, neg_f, sum_Q, total_count, total_time, emb_density);
    }
}

// Compute non-edge forces using Barnes-Hut algorithm (original t-SNE)
void DA_SPTree::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[],
				     double* sum_Q, int& total_count, double& total_time,
				     double& emb_density)
{
  double tol = 1e-5;
    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;
    
    // Compute distance between point and center-of-mass
    double D = .0;
    unsigned int ind = point_index * dimension;
    for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
    
    // Check whether we can use this node as a "summary"
    double max_width = 0.0;
    double cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
        cur_width = boundary->getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }
    if(is_leaf || max_width / sqrt(D) < theta) {
      double dist = sqrt(D); 
        // Compute and add t-SNE force between point and current node
      clock_t start = clock();
        D = 1.0 / (1.0 + D);
        double mult = cum_size * D;
        *sum_Q += mult;
	emb_density += mult*log(dist+tol);
	// emb_density += mult; 
        mult *= D;
	clock_t end = clock(); 
        for(unsigned int d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
	total_time += (float) (end - start) / CLOCKS_PER_SEC;
	total_count++; 
    }
    else {

        // Recursively apply Barnes-Hut to children
      for(unsigned int i = 0; i < no_children; i++) children[i]->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, total_count, total_time, emb_density);
    }
}


// Computes edge forces
void DA_SPTree::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f, bool lying, bool density, double* emb_densities, double* val_D)
{
    
    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    unsigned int ind2 = 0;
    double D;
    double nu = 1.;
    double tol = 1e-5;
    
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
	  if (lying) { 
	    // nu = (1. + atan(betas[n] + betas[col_P[i]]));

	  // nu = 1.;
	    nu = 1 + betas[n] + betas[col_P[i]];
	  }
	  if (density) {
	    emb_densities[n] += val_P[i] * log(val_D[i] + tol); 
	  }
	  // Need to sum Ps don't we?

	  
            // Compute pairwise distance and Q-value
            D = 1.0;
            ind2 = col_P[i] * dimension;
            for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind1 + d] - data[ind2 + d];
            for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d]/nu;
            D = val_P[i] / D;
	    // D = val_P[i] * pow(D, -(nu+1)/2.); 
	    // printf("[%f, {%f, %f}]; ", D, buff[0], buff[1]); 
            // Sum positive force
            for(unsigned int d = 0; d < dimension; d++) {
	      pos_f[ind1 + d] += (nu+1)/nu * D * buff[d];
	    }
	    
	    // for(unsigned int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * buff[d];
        }
        ind1 += dimension;
    }
}


// Print out tree
void DA_SPTree::print() 
{
    if(cum_size == 0) {
        printf("Empty node\n");
        return;
    }

    if(is_leaf) {
        printf("Leaf node; data = [");
        for(int i = 0; i < size; i++) {
            double* point = data + index[i] * dimension;
            for(int d = 0; d < dimension; d++) printf("%f, ", point[d]);
            printf(" (index = %d)", index[i]);
            if(i < size - 1) printf("\n");
            else printf("]\n");
        }        
    }
    else {
        printf("Intersection node with center-of-mass = [");
        for(int d = 0; d < dimension; d++) printf("%f, ", center_of_mass[d]);
        printf("]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print();
    }
}

