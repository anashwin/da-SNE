#ifndef CELL_H
#define CELL_H 

using namespace std;

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

#endif
