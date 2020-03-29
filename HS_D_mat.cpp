#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"
#include "GreenAnisotropic2D.hpp"

//#include <iostream>

//double heterogeneous_medium::D_global(int s, int r, int I, int J) {
//	return M_global(s,r,I,J)+T_global(s,r,I,J);
//}

std::vector<std::vector<double>> heterogeneous_medium::D_mat0_assemble(int self_flag) {
	//
	int size_Dmat0=3*n_al;
	std::vector<std::vector<double>> Dmat0(size_Dmat0,std::vector<double>(size_Dmat0,0.));
	//
	for (int i=0;i<size_Dmat0;++i) {
		for (int j=0;j<size_Dmat0;++j) {
			Dmat0[i][j]=M_global(0,0,i,j)+T_global(0,0,i,j,self_flag);
		}
	}
	return Dmat0;
}

std::vector<std::vector<double>> heterogeneous_medium::D_mat_assemble(int self_flag) {
	//
	std::vector<int> start_inds(p+1);
	//
	std::vector<int> D_sizes(p+1);
	//
	for (int r=1;r<p+1;++r) {
		start_inds[r]=3*n_al*((r-1)*(r-1)+3*(r-1))/2;
		D_sizes[r]=3*n_al*(r+1);
	}
	//
	int size_Dmat=3*n_al*(p*p+3*p)/2;
	std::vector<std::vector<double>> Dmat(size_Dmat,std::vector<double>(size_Dmat,0.));
	//
	for (int r=1;r<p+1;++r) {
		for (int s=1;s<p+1;++s) {
			for (int i=0;i<D_sizes[r];++i) {
				for (int j=0;j<D_sizes[s];++j) {
					Dmat[start_inds[r]+i][start_inds[s]+j]=M_global(s,r,i,j)+T_global(s,r,i,j,self_flag);
				}
			}
		}
	}
	return Dmat;
}
