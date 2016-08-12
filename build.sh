if [[ $(which module) ]]; then
    module load CUDA/7.5.18
fi

nvcc -std=c++11 -O2 *.cpp -lcufft -lcufftw
