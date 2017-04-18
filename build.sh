unset SM_ARCH
if [[ $(uname -n | grep build) ]]; then
#    module load GCC/5.4.0
#    module load CUDA/8.0.61
    module load GCC/4.9.2
    module load CUDA/7.5.18
else
    echo "Make sure you have CUDA toolchain installed."
    SM_ARCH="-arch sm_35"
fi

echo "Building... (This may take a few minutes.)"
if [[ $1 == "debug" ]]; then
    nvcc -DDEBUG $SM_ARCH -o pspectre-debug -g -std=c++11 -O0 *.cpp *.cu -lcufft -lcufftw
elif [[ $1 == "debug-quiet" ]]; then
    nvcc $SM_ARCH -o pspectre-debug -g -std=c++11 -O0 *.cpp *.cu -lcufft -lcufftw
else
    nvcc -std=c++11 $SM_ARCH -o pspectre -O2 *.cpp *.cu -lcufft -lcufftw
fi
