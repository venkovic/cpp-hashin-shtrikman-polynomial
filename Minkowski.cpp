#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"

double heterogeneous_medium::W(int r, int ni1, int ni2) {
	// ni1+ni2=r
	double a=L/2.;
	double Wr_i=(pow(a/2.,ni1+ni2+2)-pow(-a/2.,ni1+1)*pow(a/2.,ni2+1)-pow(a/2.,ni1+1)*pow(-a/2.,ni2+1)+pow(-a/2.,ni1+1)*pow(-a/2.,ni2+1))/(ni1+1)/(ni2+1);
	return Wr_i;
}


