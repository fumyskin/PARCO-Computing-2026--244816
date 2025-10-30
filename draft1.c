#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"


typedef struct {
    int n_rows;
    int n_cols;
    int nnz;
    int* row_indices;
    int* col_indices;
    double* values;
}Sparse_Coordinate;

typedef_struct{
    int n_rows;
    int n_cols;
    int nnz;
    int* row_indices;
    int* col_indices;
    double* values;
}Sparse_CSR;

//function to initialize a struct COO given the data extracted from .mtx file
Sparse_Coordinate* initialize_COO(
    int n_rows,
    int n_cols,
    int nnz,
    int* row_indices,
    int* col_indices,
    double* values
)
{
    Sparse_Coordinate* struct_COO = malloc(sizeof(Sparse_Coordinate));
    struct_COO->n_rows = n_rows;
    struct_COO->n_cols = n_cols;
    struct_COO->nnz = nnz;
    struct_COO->row_indices = row_indices;
    struct_COO->col_indices = col_indices;
    struct_COO->values = values;

    return struct_COO;
}

//function to perform matrix
void SpMV_COO(Sparse_Coordinate* COO, double* vec, double* res){

    for(int i = 0; i < COO->n_rows; i++){
        res[i] = 0;
    }

    for(ssize_t nnz_id = 0; nnz_id < COO->nnz; nnz_id++){
        ssize_t i = COO->row_indices[nnz_id];
        ssize_t j = COO->col_indices[nnz_id];
        double val = COO->values[nnz_id];

        res[i] += val * vec[j]; 
    } 

    return;
}






int main(int argc, char *argv[])
{
    //FOR NOW I'LL USE THE EXAMPLE GIVEN
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double *val;

    /*Initialize struct for sparse matrix */
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++){
        fprintf(stdout, "%d %d %20.19g\n", I[i]+1, J[i]+1, val[i]);
    }

    //create struct with data read from .mtx file
    Sparse_Coordinate* struct_COO = initialize_COO(M, N, nz, I, J, val);

    //INITIALIZE MATRIX VECTOR MULTIPLICATION
    double* res = malloc(N * sizeof(double));
    double* vec = malloc(M * sizeof(double));

    srand(0);
    for(int i = 0; i < N; i++){
        vec[i] = rand() % 10;
    }

    //compute SpMV
    SpMV_COO(struct_COO, vec, res);

    printf("\nResult (first 10 entries):\n");
    for (int i = 0; i < M && i < 100; i++) {
        printf("res[%d] = %g\n", i, res[i]);
    }

    free(I);
    free(J);
    free(val);
    free(vec);
    free(res);
    free(struct_COO);
  
	return 0;
}





/*
NOTES ON PARALLELIZATION
CACHE

- efficient parallel code we need to assure that SEQUENTIAL PARTS ARE PERFORMANT
- TRY TO COORDINATE SEQUENTIAL PARTS in such a way that exploit cache as much as possible (eg row major, ...)
- WARNING:
    - FALSE SHARING MAY BE A PROBLEM
    -> if you use cache, pay attention to cache lines: if data is invalidated, the entire content of cache line is
    -> FORCE VARIABLES WHICH ARE ACCESSED BY DIFFERENT THREADS TO BE ON DIFFERENT CACHE LINES
    
        struct alignTo64ByteCacheLine {
            int _onCacheLine1 __attribute__((aligned(64)))
            int _onCacheLine2 __attribute__((aligned(64)))
        }

    - BRANCH PREDICTION
    -> Random data leads to unpredictable branches, slowing execution
    -> SORTING data can improve branch prediction and speed up execution

*/