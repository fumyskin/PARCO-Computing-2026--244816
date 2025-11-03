#ifndef SPECIFICATIONS_H
#define SPECIFICATIONS_H

#include<stddef.h>

typedef struct {
    unsigned n_rows;
    unsigned n_cols;
    unsigned nnz;
    unsigned *row_indices;
    unsigned *col_indices;
    double *values;
} Sparse_Coordinate;

void *surely_malloc(size_t size);
void coo_quicksort(Sparse_Coordinate *p);

#endif