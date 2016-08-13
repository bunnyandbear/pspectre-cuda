if [[ $(which module) ]]; then
    module load GCC/5.1.0
    module load CUDA/7.5.18
fi

nvcc -std=c++11 -O2 *.cpp *.cu -lcufft -lcufftw
