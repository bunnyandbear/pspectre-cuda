#include <stdio.h>

void print_cuda_info()
{
	int nr_dev = 0;

	cudaGetDeviceCount(&nr_dev);
	if (nr_dev <= 0) {
		printf("==========================\n");
		printf("WARNING! WARNING! WARNING!\n");
		printf("No CUDA device found.\n");
		printf("==========================\n");
	}
	for (int i = 0; i < nr_dev; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
		       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) /
		       1.0e6);
	}
}
