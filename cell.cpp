#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "cell.h"

/*
#ifndef CELL
#define CELL

using namespace std;
*/
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
// #endif
