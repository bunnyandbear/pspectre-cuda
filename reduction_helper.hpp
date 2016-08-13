struct double_array_gpu {
	int dimx, dimy, dimz;
	double *array;
	double_array_gpu(int x, int y, int z);
	~double_array_gpu();
	double sum();
};
