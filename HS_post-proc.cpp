#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"
#include "GreenAnisotropic2D.hpp"

#include "iostream"

std::vector<double> heterogeneous_medium::tau(std::vector<double> &tau0, std::vector<double> &tau_grads, int al, double dx1, double dx2) {
	// Get in-grain constant part of the polarization stress field
	std::vector<double> tau(3,0);
	for (int i=0;i<3;++i) {
		tau[i]=tau0[3*al+i];
	}
	//
	std::vector<int> tau_sizes(p+1,0);
	//for (int i=1;i<=p+1;++i) {
	for (int i=1;i<p+1;++i) {
		tau_sizes[i]=3*n_al*(i+1);
	}
	//
	for (int n_i=1;n_i<p+1;++n_i) {
		for (int i=0;i<n_i+1;++i) {
			int ni1, ni2;			
			if (i%2==0) { 			// i   = 0,   2,     4,     6,     ..., n_i.
				ni1=n_i-i/2; 		// ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni2=n_i-ni1;		// ni2 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			}
			else {					// i   = 1,   3,     5,     7,     ..., n_i
				ni2=n_i-(i-1)/2; 	// ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni1=n_i-ni2; 		// ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			}
			// Get number of indices combinations used for Mandel representation
			double fac_i=sqrt(Binom(n_i,ni1));
			//
			
			int i_start=0;
			for (int j=0;j<n_i;++j) {
				i_start+=tau_sizes[j];
			}
			
			tau[0]+=fac_i*tau_grads[i_start+al*(n_i+1)*3+i*3]*pow(dx1,ni1)*pow(dx2,ni2);
			tau[1]+=fac_i*tau_grads[i_start+al*(n_i+1)*3+i*3+1]*pow(dx1,ni1)*pow(dx2,ni2);
			tau[2]+=fac_i*tau_grads[i_start+al*(n_i+1)*3+i*3+2]*pow(dx1,ni1)*pow(dx2,ni2);
		}
	}
	tau[2]/=sqrt(2);
	return tau;
}


double heterogeneous_medium::error_div(std::vector<double> &tau_grads, int al, double dx1, double dx2) {
	// Initilize divergence
//	std::cout << "test_A0" << std::endl;
	std::vector<double> div_tau(2,0);
	//
//	std::cout << "test_A1" << std::endl;
	std::vector<int> tau_sizes(p+1,0);
	for (int i=1;i<p+1;++i) {
		tau_sizes[i]=3*n_al*(i+1);
	}
	//
//	std::cout << "test_A" << std::endl;
	for (int n_i=1;n_i<p+1;++n_i) {
		for (int i=0;i<n_i+1;++i) {
//			std::cout << "test_B " << n_i << " " << i << std::endl;
			int ni1, ni2;
			if (i%2==0) {				// i   = 0,   2,     4,     6,     ..., n_i.
				ni1=n_i-i/2;			// ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni2=n_i-ni1;			// ni2 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			}
			else {						// i   = 1,   3,     5,     7,     ..., n_i
				ni2=n_i-(i-1)/2; 		// ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni1=n_i-ni2;			// ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			}
			// Get number of indices combinations used for Mandel representation
			double fac_den=sqrt(Binom(n_i,ni1));
			//
			int i_start=0;
			for (int j=0;j<n_i;++j) {
				i_start+=tau_sizes[j];
			}
			//
			if (ni1>0) {
				double fac1=Binom(n_i-1,ni1-1)/fac_den;
				div_tau[0]+=n_i*fac1*tau_grads[i_start+al*(n_i+1)*3+i*3+0]*pow(dx1,ni1-1)*pow(dx2,ni2);
				div_tau[1]+=n_i*fac1*tau_grads[i_start+al*(n_i+1)*3+i*3+2]/sqrt(2.)*pow(dx1,ni1-1)*pow(dx2,ni2);
				}
			if (ni2>0) {
				double fac2=Binom(n_i-1,ni2-1)/fac_den;
				div_tau[0]+=n_i*fac2*tau_grads[i_start+al*(n_i+1)*3+i*3+2]/sqrt(2.)*pow(dx1,ni1)*pow(dx2,ni2-1);
				div_tau[1]+=n_i*fac2*tau_grads[i_start+al*(n_i+1)*3+i*3+1]*pow(dx1,ni1)*pow(dx2,ni2-1);
			}
		}
	}
	return sqrt(pow(div_tau[0],2)+pow(div_tau[1],2));
}


