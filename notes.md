
# NOTES ON PARALLELIZATION
## CACHE

- efficient parallel code we need to assure that SEQUENTIAL PARTS ARE PERFORMANT
- TRY TO COORDINATE SEQUENTIAL PARTS in such a way that exploit cache as much as possible (eg row major, ...)
- **WARNING**:
    - **FALSE SHARING MAY BE A PROBLEM**
    -> if you use cache, pay attention to cache lines: if data is invalidated, the entire content of cache line is
    -> FORCE VARIABLES WHICH ARE ACCESSED BY DIFFERENT THREADS TO BE ON DIFFERENT CACHE LINES
    
        struct alignTo64ByteCacheLine {
            int _onCacheLine1 __attribute__((aligned(64)))
            int _onCacheLine2 __attribute__((aligned(64)))
        }

    - **BRANCH PREDICTION**
    -> Random data leads to unpredictable branches, slowing execution
    -> SORTING data can improve branch prediction and speed up execution



# REFERENCES
https://ieeexplore.ieee.org/document/10444348 -> paper to Efficient COO to CSR Conversion for Accelerating Sparse Matrix Processing on FPGA

https://arxiv.org/abs/2510.13412 -> paper to Formal Verification of COO to CSR Sparse Matrix Conversion (Invited Paper) -> https://www.cs.princeton.edu/~appel/papers/coo-csr.pdf

https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c -> Convert COO to CSR format in c++ stackoverflow