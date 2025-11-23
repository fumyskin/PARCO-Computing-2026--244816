# PARCO-Computing-2026--244816
Parallel computing course repository 

# CODE AVAILABLE IN THE REPO
In this deliverable are going to be implemented the following *.c files:
- experiment0.c: the original sequential code
- experiment1.c: the first parallelized attempt with pragma OMP constructs; to change the scheduling type, simply correct the line of the outer loop with #pragma omp parallel for (schedule(...))
- experiment2.c: a manual attempt to divide workload; posix_mem_align() is used to minimize false sharing
- experiment3.c: an attempt to use #pragma task 
- experiment4.c: a fused solution between experiment1.c and experiment2.c: it tries to minimize false sharing by using cache alignment while using #pragma omp static
- experiment6.c: an attempt to implement the merge-based CSR spmvm algorithm, a state-of-the-art approach to further explore CSR spmvm parallelization

# INSTRUCTIONS TO COMPILE AND RUN THE CODE
We have two approaches to compile the testbenches:
- **WITHOUT optimization flags**: "gcc mmio.h specifications.c experiment0.c -fopenmp -o spmv_sequential"
- **WITH optimization flags**: "gcc -O3 mmio.h specifications.c experiment0.c -fopenmp -ffp-contract=fast -o spmv_sequential"

The optimization flags help with branch prediction but also make the FMA operation in the CSR spmvm faster.
The names of the executables must be in the form of "spmv_*" where *: sequential, static, dynamic, guided, auto, runtime, runtime_static, runtime_dynamic, runtime_guided, manual, bind, merge.

Here is the complete mapping of original code/executable:
1. spmv_sequential -> experiment0.c
2. spmv_static -> experiment1.c with #pragma omp parallel for schedule(static)
3. spmv_dynamic -> experiment1.c with #pragma omp parallel for schedule(dynamic)
4. spmv_guided -> experiment1.c with #pragma omp parallel for schedule(guided)
5. spmv_auto -> experiment1.c with #pragma omp parallel for schedule(auto)
6. spmv_runtime -> experiment1.c with #pragma omp parallel for schedule(runtime)
7. spmv_runtime_static -> experiment1.c with #pragma omp parallel for schedule(runtime)
8. spmv_runtime_dynamic -> experiment1.c with #pragma omp parallel for schedule(runtime)
9. spmv_runtime_guided -> experiment1.c with #pragma omp parallel for schedule(runtime)
10. spmv_manual -> experiment2.c
11. spmv_task -> experiment3.c 
12. spmv_bind -> experiment4.c, **IN TESTBENCHES IT IS RUN WITH THE compile1.sh script**
13. spmv_merge -> experiment6.c

All these executables are to be run on all matrices in the /MATRICES folder. The data are going to be run by means of 2 scripts: **compile1.sh** for experiment4.c and **compile.sh** for all the others.
The files are to be saved in a .csv file, then given in input in 2 matlab scripts that processes data and plot results. More specifically: 

- The *compile.sh* and *compile1.sh* scripts are made to be run on different number of threads, for all the given matrices in the /MATRICES dir and, if the chunksize is specified, run the *spmv_static/dynamic/guided_runtime* exectuables with the given number of chunks; ***compile1.sh* script is made specifically for *spmv_bind* as it contains the PROC_BIND commands for thread pinning.**
- If one wish to collect results for a given executable/s, just insert the name/s of the executable/s "spmv_*" in the $EXECUTABLE bash array, respecting the rules stated in the point before; one is free to adjust the threads, the chunksize and the matrices analysed by changing the other bash arrays found at the start of the script.
- 