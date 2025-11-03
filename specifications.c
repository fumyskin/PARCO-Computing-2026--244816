#include <stdlib.h>
#include <stdio.h>
#include "specifications.h"


//implement surely_malloc function
void *surely_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Out of memory while allocating %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}


// quicksort taken from Appel paper and modified for COO struct : qsort3.c; 
// https://github.com/cverified/cbench-vst/blob/master/qsort/qsort3.c

#define SWAP_UNSIGNED(a, b) do { unsigned tmp = (a); (a) = (b); (b) = tmp; } while(0)
#define SWAP_DOUBLE(a, b)  do { double tmp = (a); (a) = (b); (b) = tmp; } while(0)

/* Lexicographic quicksort by (row, col) */
static void quicksort_coo_recursive(unsigned *row, unsigned *col, double *val,
                                    int lo, int hi)
{
    if (lo >= hi)
        return;

    int i = lo;
    int j = hi;
    int mid = lo + ((hi - lo) >> 1);

    unsigned piv_row = row[mid];
    unsigned piv_col = col[mid];

    while (i <= j) {
        // Move i right while (row[i], col[i]) < (piv_row, piv_col)
        while ((row[i] < piv_row) ||
               (row[i] == piv_row && col[i] < piv_col))
            i++;

        // Move j left while (row[j], col[j]) > (piv_row, piv_col)
        while ((row[j] > piv_row) ||
               (row[j] == piv_row && col[j] > piv_col))
            j--;

        if (i <= j) {
            SWAP_UNSIGNED(row[i], row[j]);
            SWAP_UNSIGNED(col[i], col[j]);
            SWAP_DOUBLE(val[i], val[j]);
            i++;
            j--;
        }
    }

    if (lo < j)
        quicksort_coo_recursive(row, col, val, lo, j);
    if (i < hi)
        quicksort_coo_recursive(row, col, val, i, hi);
}

/* Public API */
void coo_quicksort(Sparse_Coordinate *p)
{
    if (p == NULL || p->nnz <= 1)
        return;
    quicksort_coo_recursive(p->row_indices, p->col_indices, p->values,
                            0, (int)p->nnz - 1);
}
