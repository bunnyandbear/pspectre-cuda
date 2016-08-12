g++ -Wall -O3 -march=native -fopenmp -DHAVE_PRIVATE *.cpp private/*.cpp -lfftw3 -lfftw3_omp
