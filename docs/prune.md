|        Benchmark       	|              Kernel             	| Before Prunning 	| After Prunning 	|
|:----------------------:	|:-------------------------------:	|:---------------:	|:--------------:	|
|    rodinia/backprop    	|      bpnn_layerforward_CUDA     	|       0.56      	|      0.93      	|
|       rodinia/bfs      	|              Kernel             	|       0.58      	|       0.8      	|
|     rodinia/b+tree     	|            findRangeK           	|       0.69      	|      0.89      	|
|       rodinia/cfd      	|        cuda_compute_flux        	|       0.59      	|      0.97      	|
|    rodinia/heartwall   	|             kernel              	|       0.58      	|      0.89      	|
|     rodinia/hotspot    	|          calculate_temp         	|       0.69      	|      0.93      	|
|     rodinia/huffman    	|   vlc_encode_kernel_sm64huff    	|       0.67      	|       0.9      	|
|     rodinia/kmeans     	|           kmeansPoint           	|       0.57      	|      0.94      	|
|     rodinia/lavaMD     	|         kernel_gpu_cuda         	|       0.5       	|      0.88      	|
|       rodinia/lud      	|           lud_diagonal          	|       0.68      	|      0.87      	|
|     rodinia/myocyte    	|             solver_2            	|       0.47      	|      0.93      	|
|       rodinia/nw       	|       needle_cuda_shared_1      	|       0.58      	|      0.74      	|
|     rodinia/sradv1     	|              reduce             	|       0.68      	|      0.89      	|
|   rodinia/pathfinder   	|          dynproc_kernel         	|       0.52      	|      0.88      	|
|    rodinia/gaussian    	|               Fan2              	|       0.6       	|      0.84      	|
| rodinia/particlefilter 	|        likelihood_kernel        	|       0.62      	|      0.96      	|
|  rodinia/streamcluster 	|       kernel_compute_cost       	|       0.49      	|      0.88      	|
|       Quicksilver      	|       CycleTrackingKernel       	|       0.69      	|      0.93      	|
|        ExaTENSOR       	|         tensor_transpose        	|       0.68      	|      0.92      	|
