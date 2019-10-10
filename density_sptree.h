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


#ifndef SPTREE_H
#define SPTREE_H

using namespace std;
/*
class Cell {

    unsigned int dimension;
    double* corner;
    double* width;
    
    
public:
    Cell(unsigned int inp_dimension);
    Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width);
    ~Cell();
    
    double getCorner(unsigned int d);
    double getWidth(unsigned int d);
    void setCorner(unsigned int d, double val);
    void setWidth(unsigned int d, double val);
    bool containsPoint(double point[]);
};
*/
class Cell;

class SPTree
{
    
    // Fixed constants
    static const unsigned int QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    double* buff;
    
    // Properties of this node in the tree
    SPTree* parent;
    unsigned int dimension;
    bool is_leaf;
    unsigned int size;
    unsigned int cum_size;
        
    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell* boundary;
    
    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
    double* data;
    double* center_of_mass;
    unsigned int index[QT_NODE_CAPACITY];

    // Density specific terms
    double* all_emb_dens; 
    double* all_log_emb_dens;  // normalized to mean zero
    double* all_log_orig_dens;  // normalized to mean zero, variance 1
    double* all_marg_Q; // marginals at each point 

    double emb_density_com; 
    double log_emb_density_com;
    double log_orig_density_com;
    double marg_Q_com; 

    // Children
    SPTree** children;
    unsigned int no_children;
    
public:
    SPTree(unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, unsigned int N);
    SPTree(unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, double* inp_corner, double* inp_width);
    SPTree(unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, unsigned int N, double* inp_corner, double* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, unsigned int N, double* inp_corner, double* inp_width);
    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, double* inp_corner, double* inp_width);
    ~SPTree();
    void setData(double* inp_data);
    SPTree* getParent();
    void construct(Cell boundary);
    bool insert(unsigned int new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
    void getAllIndices(unsigned int* indices);
    unsigned int getDepth();
    void computeDensityForces(unsigned int point_index, double theta, double dense_f1[], 
			      double dense_f2[], bool verbose=false); 
    void print();
    
private:
    void init(SPTree* inp_parent, unsigned int D, double* inp_data, double* emb_densities, double* log_emb_densities, 
	   double* log_orig_densities, double* marg_Q, double* inp_corner, double* inp_width);
    void fill(unsigned int N);
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
    bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
};

#endif
