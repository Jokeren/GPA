# An introduction to gpa.advice

gpa.advice is generated under `gpa-database` for each benchmark. Currently, we provide a text format report and use hpcviewer to visualize blames on individual lines of GPU kernels with its CPU calling context.

## Format

## Benchmarks

### rodinia/backprop
 - **Kernel**: `bpnn_layerforward_CUDA`
 - **Version**: origin
 - **Estimate Speedup**: 1.15x
 - **Achieved Speedup**: 1.15x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUWarpBalance optimizer suggests removing synchronizations on several lines. We remove the synchronization on Line 35, which contributes to 1.15x speedup as projected. Due the the program's constraint, other synchronizations cannot be removed.

 - **Kernel**: `bpnn_layerforward_CUDA`
 - **Version**: opt1
 - **Estimate Speedup**: 1.10x
 - **Achieved Speedup**: 1.21x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPUStrengthReduction optimizer indicates inlined division functions on Line 40 cause high stalls. Using the `fast_math` flag does not help because SFU instructions are still low throughput and high latency. Therefore, we use cheap bit-wise operations to replace integer divisions.

### rodinia/bfs
 - **Kernel**: `Kernel`
 - **Version**: origin
 - **Estimate Speedup**: 1.58x
 - **Achieved Speedup**: 1.10x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPULoopUnrolling optimizer suggests unrolling the loop at Line 28. We manually unroll the loop by a factor of four to hide latencies. There is a large gap between the achieved speedup and the estimate speedup. In this case, we found that the workload is highly unbalanced such that most threads only execute less than four iterations of the loop. Thus, loop unrolling benefits only a small number of threads.

### rodinia/b+tree
 - **Kernel**: `findRangeK`
 - **Version**: origin
 - **Estimate Speedup**: 1.28x
 - **Achieved Speedup**: 1.16x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPUCodeReorder optimizer suggests the indirect memory accesses on Line 62 and Line 29 can be decomposed to reorder instructions.

### rodinia/cfd
 - **Kernel**: `cuda_compute_flux`
 - **Version**: origin
 - **Estimate Speedup**: 1.60x
 - **Achieved Speedup**: 1.68x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUCodeReorder optimizer suggests reordering memory load instructions to hide latency. And the #2 GPUFastMath optimizer suggests using fast math functions. Note that in this code reorder itself helps only little because math functions are not inlined such that some barrier dependencies are resolved before the functions are called. We enable `--use_fast_math` to inline math functions (and reduce instructions), letting the compilers to reorder instructions. Thus, the estimate speedup is a combination of GPUCodeReoder and GPUFastMath optimizers.

### rodinia/gaussian
 - **Kernel**: `Fan2`
 - **Version**: origin
 - **Estimate Speedup**: 3.30x
 - **Achieved Speedup**: 3.32x
 - **Section**: Parallel Optimizer
 - **Description**:
 
The #1 GPUOccupancyIncrease optimizer suggests increasing the number of threads to increase occupancy.

### rodinia/heartwall
 - **Kernel**: `kernel`
 - **Version**: origin
 - **Estimate Speedup**: 1.17x
 - **Achieved Speedup**: 1.18x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPULoopUnroll optimizer suggests unrolling two nested loops. We use `pragma unroll` for the outer loop which has an estimate speedup of 1.17x. Unroll the inner loop does not achieve the estimate 1.14x by GPA with NVCC-11.0. However, NVCC-11.2 does a better job in loop unrolling and and resulting in 1.33x speedup, which is close to our estimate.

### rodinia/hotspot
 - **Kernel**: `calculate_temp`
 - **Version**: origin
 - **Estimate Speedup**: 1.13x
 - **Achieved Speedup**: 1.60x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPUStrengthReduction optimizer suggests replacing expensive divisions with reciprocal.

### rodinia/huffman
 - **Kernel**: `vlc_encode_kernel_sm64huff`
 - **Version**: origin
 - **Estimate Speedup**: 1.19x
 - **Achieved Speedup**:  1.07x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUWarpBalance optimizer points out several unbalanced regions. We replace the top one on Line 108 that performs synchronization within warps with syncwarp.

### rodinia/kmeans
 - **Kernel**: `kmeansPoint`
 - **Version**: origin
 - **Estimate Speedup**: 1.11x
 - **Achieved Speedup**: 1.20x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPULoopUnrolling optimizer suggests unrolling the loop at Line 86.
 
### rodinia/lavaMD
 - **Kernel**: `kernel_gpu_cuda`
 - **Version**: origin
 - **Estimate Speedup**: 1.11x
 - **Achieved Speedup**: 1.12x
 - **Section**: Code Optimizer
 - **Description**:
 
The #2 GPULoopUnrolling optimizer suggests unrolling the loop at Line 145 which has a large constant trip count. Note the actual benefits of unrolling may be diminished with a newer version of NVCC which does better auto unrolling.

### rodinia/lud
 - **Kernel**: `lud_diagonal`
 - **Version**: origin
 - **Estimate Speedup**: 1.41x
 - **Achieved Speedup**: 1.40x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUCodeReorder optimizer indicates high WAR and SMEM latencies, which implies high register dependencies. We note data are fetched from the shared memory in every iteration. Hence, we can use a temporary variable to store and update values. Optimizing the top three hot regions in total leads to a 1.41x speedup. Because regions are not overlapped. the overall speedup is a little bit higher than the sum of speedups from different regions.

### rodinia/myocyte
 - **Kernel**: `solver_2`
 - **Version**: origin
 - **Estimate Speedup**: 1.22x
 - **Achieved Speedup**: 1.13x
 - **Section**: Code Optimizer
 - **Description**:
 
Though there are a list of optimizers pointing of promising optimizing spots, the kernel contains large and complicated code. Instead of manually adjusting lines, we adopt the #6 GPUFastMath optimizer to enable fast math functions.

 - **Kernel**: `solver_2`
 - **Version**: opt1
 - **Estimate Speedup**: 1.01x
 - **Achieved Speedup**: 1.02x
 - **Section**: Code Optimizer
 - **Description**:
 
Because of the large kernel that contains a bunch of GPU device functions, the GPUFunctionSplit optimizer ranks the fourth in the list. It is worth noting that not every inlined function can or should be split from the inline site, since the benefits of reducing instruction cache miss might be less than the overhead of increasing the number of instructions. In practice, we only find the third hot function (`kernel_cam_2`) has actual benefits.

### rodinia/nw
 - **Kernel**: `needle_cuda_shared_1`
 - **Version**: origin
 - **Estimate Speedup**: 1.07x
 - **Achieved Speedup**: 1.10x
 - **Section**: Code Optimizer
 - **Description**:
 
Based on the #3 GPUWarpBalance optimizer, we have a list of hot synchronization instructions. Among them, we can safely erase Line 134 and Line 142.

### rodinia/particlefilter
 - **Kernel**: `likelihood_kernel`
 - **Version**: origin
 - **Estimate Speedup**: 2.0x
 - **Achieved Speedup**: 1.75x
 - **Section**: Parallel Optimizer
 - **Description**:
 
The GPUBlockIncrease optimizer suggests adjusting the number of threads per block. The profiling results shows we used two blocks, which takes 1/40 SMs of a V100 GPU. We reduce the number of threads per block from 512 to 256 to use 1/20 SMs of a GPU.

### rodinia/streamcluster
 - **Kernel**: `kernel_compute_cost`
 - **Version**: origin
 - **Estimate Speedup**: 1.52x
 - **Achieved Speedup**: 1.31x
 - **Section**: Parallel Optimizer
 - **Description**:
 
Similar to particlefilter, we increase the number of blocks according to the GPUBlockIncrease optimizer.

### rodinia/sradv1
 - **Kernel**: `reduce`
 - **Version**: origin
 - **Estimate Speedup**: 1.08x
 - **Achieved Speedup**: 1.03x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUWarpBalance optimizer shows a list of expansive synchronization instructions. The kernel adopts a reduce pattern to sum values among all the threads. On Line 64, we reduce the number of synchronizations by not performing reduce in the last few steps of reduction and instead let thread 1 sums up all values directly. 

### rodinia/pathfinder
 - **Kernel**: `dynproc_kernel`
 - **Version**: origin
 - **Estimate Speedup**: 1.35x
 - **Achieved Speedup**: 1.04x
 - **Section**: Code Optimizer
 - **Description**:
 
The #1 GPUCodeReorder optimizer suggests reordering a global memory read in a loop of the pathfinder benchmark. The estimated speedup is 30% higher than we achieved because instructions after synchronizations depend on the results before synchronizations. Therefore, the instructions we can use to hide latency are limited in a fine-grained scope in which the distance between the dependent instruction pairs is short no matter how we arrange instructions.

## Visualization of instruction blames
