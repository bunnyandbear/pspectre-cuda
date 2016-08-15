#include <cufftw.h>
#include <iostream>
#include <cstdlib>

#include "field_size.hpp"

class host_field {
	fftw_complex *ptr;
public:
	host_field(field_size &fs) {
		ptr = (fftw_complex *) malloc(fs.alloc_size);
		if (ptr == NULL) {
			std::cout << "host_field: malloc failed" << std::endl;
			exit(1);
		}
	}
	~host_field() {
		free(ptr);
	}
	operator fftw_complex *() {
		return ptr;
	}
	double *data() {
		return (double *) ptr;
	}
};
