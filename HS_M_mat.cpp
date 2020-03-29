#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"
#include "GreenAnisotropic2D.hpp"


//#include <iostream>

double heterogeneous_medium::M_global(int s, int r, int I, int J) {
	//std::vector<std::vector<double>> dMW(3*(r+1)*n_al,std::vector<double>(3*(s+1)*n_al,0.));
	int ddim_i=3*(r+1);
	int ddim_j=3*(s+1);
	//
	int ki=I%ddim_i;
	int k=(I-ki)/ddim_i;
	int kj=J%ddim_j;
	int kk=(J-kj)/ddim_j;
	//
	if (kk==k) {
		if ((r==0)&&(s==0)) {
			return cfrac[k]*dM_mandel[k][ki][kj];

		}
		else {
			return dMW_local(k,s,r,ki,kj);
		}
	}
	//
	return 0.;
}

double heterogeneous_medium::dMW_local(int al, int s, int r, int I, int J) {
	//std::vector<std::vector<double>> dMW(3*(r+1),std::vector<double>(3*(s+1),0.));
	//
	int ki=I%3;
	int i=(I-ki)/3;
	int kj=J%3;
	int j=(J-kj)/3;
	//
	int ni1, ni2;
	if (i%2==0) {	 		// i   = 0, 2,   4,   6,   ..., r
		ni1=r-i/2;   	// ni1 = r, r-1, r-2, r-3, ..., (r+1)/2
		ni2=r-ni1;		// ni2 = 0, 1,   2,   3,   ..., r-(r+1)/2
	}
	else {				// i   = 1, 3,   5,   7,   ..., r
		ni2=r-(i-1)/2;  // ni2 = r, r-1, r-2, r-3, ..., (r+1)/2
		ni1=r-ni2;      // ni1 = 0, 1,   2,   3,   ..., r-(r+1)/2
	}
	// Get number of indices combinations used for Mandel representation
	double fac_i=sqrt(Binom(r,ni1));
	//
	int nj1, nj2;		
	if (j%2==0) {	 		// j   = 0, 2,   4,   6,   ..., s
		nj1=s-j/2;   	// nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
		nj2=s-nj1;		// nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
	}
	else { 					// j   = 0, 2,   4,   6,   ..., s
		nj2=s-(j-1)/2;  // nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
		nj1=s-nj2;		// nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
	}
	// Get number of indices combinations used for Mandel representation
	double fac_j=sqrt(Binom(s,nj1));
	//
	// Populate local Mandel matrix of Minkowski-weigthed compliance differential
	return fac_i*fac_j*W(r+s,ni1+nj1,ni2+nj2)*dM_mandel[al][ki][kj];
}