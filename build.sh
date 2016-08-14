if [[ $(uname -n | grep build-sb) ]]; then
    module load GCC/4.9.2
    module load CUDA/7.5.18
else
    echo "Make sure you have CUDA toolchain installed."
fi

echo "Building... (This may take a few minutes.)"
if [[ $1 == "debug" ]]; then
    nvcc -DDEBUG -arch sm_35 -o pspectre-debug -g -std=c++11 -O0 *.cpp *.cu -lcufft -lcufftw --compiler-options -fopenmp
else
    nvcc -std=c++11 -arch sm_35 -o pspectre -O2 *.cpp *.cu -lcufft -lcufftw --compiler-options -fopenmp
fi
