#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"
#include "GreenAnisotropic2D.hpp"

#include <iostream>

void heterogeneous_medium::set_mat(double L_in, int n_al_in, std::vector<std::vector<double>> xloc_in, std::vector<double> cfrac_in, std::vector<std::vector<std::vector<double>>> dM_in) {
	L=L_in;
	n_al=n_al_in;
	xloc=xloc_in;
	cfrac=cfrac_in;
	dM_mandel=dM_in;
}

void heterogeneous_medium::set_ref(int sym, std::vector<double> params) {
	mat0_Green=medium();
	mat0_Green.set_sym(sym,params);
	mat0_Green.get_H();
	mat0_Green.get_S();
	//
	std::vector<std::vector<std::vector<std::vector<double>>>> vec_alpha;
	std::vector<std::vector<std::vector<double>>> vec_gamma;
	std::vector<std::vector<double>> vec_ijkl;
	std::vector<double> vec_k;
	//
	for (int alpha=0;alpha<n_al-1;++alpha) {
		for (int gamma=alpha+1;gamma<n_al;++gamma) {
			std::vector<double> dx_gamma_alpha=min_diff(xloc[gamma],xloc[alpha],L);
			double dx_r=sqrt(pow(dx_gamma_alpha[0],2)+pow(dx_gamma_alpha[1],2));
			double dx_th=atan2(dx_gamma_alpha[1],dx_gamma_alpha[0]);
			for (unsigned int i_ijkl=0;i_ijkl<list_of_ijkl.size();++i_ijkl) {
				for (int k=0;k<n+1;++k) {
					for (int i1=0;i1<k+1;++i1) {
						std::vector<int> ijkl=list_of_ijkl[i_ijkl];
						std::vector<int> v1(i1,1);
						std::vector<int> v2(k-i1,2);
						ijkl.insert(ijkl.end(), v1.begin(), v1.end());
						ijkl.insert(ijkl.end(), v2.begin(), v2.end());
						vec_k.push_back(dnGami(mat0_Green,k,ijkl,dx_r,dx_th));
					}
					vec_ijkl.push_back(vec_k);
					vec_k.clear();
				}
				vec_gamma.push_back(vec_ijkl);
				vec_ijkl.clear();
			}
			vec_alpha.push_back(vec_gamma);
			vec_gamma.clear();
		}
		dG_table.push_back(vec_alpha);
		vec_alpha.clear();
	}
}

void heterogeneous_medium::set_p(int p_in) {
	p=p_in;
}

void heterogeneous_medium::set_n(int n_in) {
	n=n_in;
}

std::vector<double> heterogeneous_medium::eps_global0(std::vector<double> eps_av_mandel) {
	std::vector<double> eps(3*n_al,0.);
	for (int al=0;al<n_al;++al) {
		eps[3*al]=cfrac[al]*eps_av_mandel[0];
		eps[3*al+1]=cfrac[al]*eps_av_mandel[1];
		eps[3*al+2]=cfrac[al]*eps_av_mandel[2];
	}
	return eps;
}

std::vector<double> heterogeneous_medium::eps_global(std::vector<double> eps_av_mandel) {
	int size_eps=0;
	for (int i=2;i<p+2;++i) {
		size_eps+=i;
	}
	size_eps*=n_al*3;
	//
	std::vector<double> eps(size_eps,0.);
	//
	std::vector<int> eps_sizes(p+1,0);
	for (int i=1;i<p+1;++i) {
		eps_sizes[i]=3*n_al*(i+1);
	}
	//	
	for (int n_i=1;n_i<p+1;++n_i) {
		for (int al=0;al<n_al;++al) {
			for (int i=0;i<n_i+1;++i) {
				int ni1, ni2;
				if (i%2==0) {			// i   = 0,   2,     4,     6,     ..., n_i
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
					i_start+=eps_sizes[j];
				}
				//			
				eps[i_start+al*(n_i+1)*3+i*3]=fac_i*W(n_i,ni1,ni2)*eps_av_mandel[0];
				eps[i_start+al*(n_i+1)*3+i*3+1]=fac_i*W(n_i,ni1,ni2)*eps_av_mandel[1];
				eps[i_start+al*(n_i+1)*3+i*3+2]=fac_i*W(n_i,ni1,ni2)*eps_av_mandel[2];
			}
		}
	}
	return eps;
}
