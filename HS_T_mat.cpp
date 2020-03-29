#include <math.h>
#include <vector>
#include "HSPolynomial.hpp"
#include "GreenAnisotropic2D.hpp"

//#include <iostream>


double heterogeneous_medium::T_global(int s, int r, int I, int J, int self_flag) {
	int ddim_i=3*(r+1);
	int ddim_j=3*(s+1);
	int ki=I%ddim_i;
	int alpha=(I-ki)/ddim_i;
	int kj=J%ddim_j;
	int gamma=(J-kj)/ddim_j;
	//
	if (self_flag==0) {
		if (gamma==alpha) {
			double T_val=0.;
			for (int gamma=0;gamma<n_al;++gamma) {
				if (alpha!=gamma) {
					T_val-=T_local(gamma,alpha,s,r,ki,kj);
				}
			}
			return T_val;
		}
		else {
			return T_local(gamma,alpha,s,r,ki,kj);
		}
	}
	else if (self_flag==1) {
		return T_local(gamma,alpha,s,r,ki,kj);
	}
	//		
	return 0.;
}


double heterogeneous_medium::T_local(int gamma, int alpha, int s, int r, int I, int J) {
	int ki=I%3;
	int i=(I-ki)/3;
	int kj=J%3;
	int j=(J-kj)/3;
	//
	std::vector<double> fac_ij={1.,1.,sqrt(2.)};
	//
	int nr1, nr2;	
	if (i%2==0) {				// i   = 0, 2,   4,   6,   ..., r
		nr1=r-i/2;   			// nr1 = r, r-1, r-2, r-3, ..., (r+1)/2
		nr2=r-nr1;				// nr2 = 0, 1,   2,   3,   ..., r-(r+1)/2
	}
	else {						// i   = 1, 3,   5,   7,   ..., r
		nr2=r-(i-1)/2;  		// nr2 = r, r-1, r-2, r-3, ..., (r+1)/2
		nr1=r-nr2;    			// nr1 = 0, 1,   2,   3,   ..., r-(r+1)/2
	}
	// Get number of indices combinations used for Mandel representation
	double fac_r=sqrt(Binom(r,nr1));
	//
	int ns1, ns2;
	if (j%2==0) {		// j   = 0, 2,   4,   6,   ..., s
		ns1=s-j/2;   	// ns1 = s, s-1, s-2, s-3, ..., (s+1)/2
		ns2=s-ns1;		// ns2 = 0, 1,   2,   3,   ..., s-(s+1)/2
	}
	else { 				// j   = 0, 2,   4,   6,   ..., s
		ns2=s-(j-1)/2;	// ns1 = s, s-1, s-2, s-3, ..., (s+1)/2
		ns1=s-ns2;		// ns2 = 0, 1,   2,   3,   ..., s-(s+1)/2
	}
	// Get number of indices combinations used for Mandel representation
	double fac_s=sqrt(Binom(s,ns1));
	//
	if (gamma!=alpha){
		return fac_r*fac_ij[ki]*fac_ij[kj]*fac_s*T_infl(gamma,alpha,ns1,ns2,list_of_ijkl_inds[ki][kj],nr1,nr2);
	}
	else {
		//
		// Compute (r,t) given by minimum difference between centers of gravity of {alpha} and {gamma}
		if (gamma<n_al-1) {
			alpha=gamma+1;
		}
		else {
			alpha=gamma-1;
		}
		std::vector<double> dx_gamma_alpha=min_diff(xloc[alpha],xloc[gamma],L);
		//
		// Inevstigate symmetry
		//std::cout <<         T_self_infl(gamma,alpha,ns1,ns2,list_of_ijkl_inds[ki][kj],nr1,nr2,dx_gamma_alpha[0],dx_gamma_alpha[1]); 
		//std::cout << ", " << T_self_infl(gamma,alpha,nr1,nr2,list_of_ijkl_inds[ki][kj],ns1,ns2,dx_gamma_alpha[0],dx_gamma_alpha[1]) << std::endl;
		//
		return fac_r*fac_ij[ki]*fac_ij[kj]*fac_s*T_self_infl(gamma,alpha,ns1,ns2,list_of_ijkl_inds[ki][kj],nr1,nr2,dx_gamma_alpha[0],dx_gamma_alpha[1]);
	}
	return 0.;
}

double heterogeneous_medium::T_infl(int alpha, int gamma, int nr1, int nr2, int ijkl_ind, int ns1, int ns2) {
	int r=nr1+nr2;
	int s=ns1+ns2;
	//
	double dkG_ijkl;
	if (gamma>alpha) {
		dkG_ijkl=dG_table[alpha][gamma-alpha-1][ijkl_ind][0][0];
		}
	else {
		dkG_ijkl=dG_table[gamma][alpha-gamma-1][ijkl_ind][0][0];
	}
	//
	//double T_i=cfrac[alpha]*cfrac[gamma]*W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2);
	double T_i=W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2)/(L*L);
	//
	for (int k=1;k<n+1;++k) {
		for (int i=0;i<k+1;++i) {
			for (int i_alpha=0;i_alpha<k-i+1;++i_alpha) {
				int n1_alpha, n2_alpha;
				if (i_alpha%2==0) {					// 0, 2,   4,   6,   ..., i
					n1_alpha=(k-i)-i_alpha/2;   	// i, i-1, i-2, i-3, ..., (i+1)/2
					n2_alpha=(k-i)-n1_alpha;		// 0, 1,   2,   3,   ..., i-(i+1)/2
				}
				else {								// 1, 3,   5,   7,   ..., i
					n2_alpha=(k-i)-(i_alpha-1)/2; 	// i, i-1, i-2, i-3, ..., (i+1)/2
					n1_alpha=(k-i)-n2_alpha; 		// 0, 1,   2,   3,   ..., i-(i+1)/2
				}				
				double fac_alpha=Binom(k-i,n1_alpha);
				//
				for (int i_gamma=0;i_gamma<i+1;++i_gamma) {
					int n1_gamma, n2_gamma;
					if (i_gamma%2==0) {				// 0, 2,   4,   6,   ..., i
						n1_gamma=i-i_gamma/2;   	// i, i-1, i-2, i-3, ..., (i+1)/2
						n2_gamma=i-n1_gamma;		// 0, 1,   2,   3,   ..., i-(i+1)/2
					}
					else {							// 1, 3,   5,   7,   ..., i
						n2_gamma=i-(i_gamma-1)/2; 	// i, i-1, i-2, i-3, ..., (i+1)/2
						n1_gamma=i-n2_gamma; 		// 0, 1,   2,   3,   ..., i-(i+1)/2
					}
					double fac_gamma=Binom(i,n1_gamma);
					//
					double dkG_ijklk1_kk;
					if (gamma>alpha) {
						dkG_ijklk1_kk=pow(-1,k)*dG_table[alpha][gamma-alpha-1][ijkl_ind][k][n1_alpha+n1_gamma];
					}
					else {
						dkG_ijklk1_kk=dG_table[gamma][alpha-gamma-1][ijkl_ind][k][n1_alpha+n1_gamma];
					}
					//
					T_i+=1./(L*L)*pow(-1,i)/Factorial(k-i)/Factorial(i)\
								 *dkG_ijklk1_kk\
					             *fac_alpha*W(r+k-i,nr1+n1_alpha,nr2+n2_alpha)\
					             *fac_gamma*W(s+i,ns1+n1_gamma,ns2+n2_gamma);
				}
			}
		}
	}
	return T_i;
}


double heterogeneous_medium::T_self_infl(int alpha, int gamma, int nr1, int nr2, int ijkl_ind, int ns1, int ns2, double dx_ga_al_1, double dx_ga_al_2) {
	int r=nr1+nr2;
	int s=ns1+ns2;
	//
	double dkG_ijkl;
	if (gamma>alpha) {
		dkG_ijkl=dG_table[alpha][gamma-alpha-1][ijkl_ind][0][0];
		}
	else {
		dkG_ijkl=dG_table[gamma][alpha-gamma-1][ijkl_ind][0][0];
	}
	//
	//double T_i=cfrac[alpha]*cfrac[gamma]*W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2);
	double T_i=W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2)/(L*L);
	//
	for (int k=1;k<n+1;++k) {
		for (int i=0;i<k+1;++i) {
			for (int i_alpha=0;i_alpha<k-i+1;++i_alpha) {
				int n1_alpha, n2_alpha;
				if (i_alpha%2==0) {					// 0, 2,   4,   6,   ..., i
					n1_alpha=(k-i)-i_alpha/2;   	// i, i-1, i-2, i-3, ..., (i+1)/2
					n2_alpha=(k-i)-n1_alpha;		// 0, 1,   2,   3,   ..., i-(i+1)/2
				}
				else {								// 1, 3,   5,   7,   ..., i
					n2_alpha=(k-i)-(i_alpha-1)/2; 	// i, i-1, i-2, i-3, ..., (i+1)/2
					n1_alpha=(k-i)-n2_alpha; 		// 0, 1,   2,   3,   ..., i-(i+1)/2
				}				
				double fac_alpha=Binom(k-i,n1_alpha);
				//
				for (int i_gamma=0;i_gamma<i+1;++i_gamma) {
					int n1_gamma, n2_gamma;
					if (i_gamma%2==0) {				// 0, 2,   4,   6,   ..., i
						n1_gamma=i-i_gamma/2;   	// i, i-1, i-2, i-3, ..., (i+1)/2
						n2_gamma=i-n1_gamma;		// 0, 1,   2,   3,   ..., i-(i+1)/2
					}
					else {							// 1, 3,   5,   7,   ..., i
						n2_gamma=i-(i_gamma-1)/2; 	// i, i-1, i-2, i-3, ..., (i+1)/2
						n1_gamma=i-n2_gamma; 		// 0, 1,   2,   3,   ..., i-(i+1)/2
					}
					double fac_gamma=Binom(i,n1_gamma);
					//
					double dkG_ijklk1_kk;
					if (gamma>alpha) {
						dkG_ijklk1_kk=pow(-1,k)*dG_table[alpha][gamma-alpha-1][ijkl_ind][k][n1_alpha+n1_gamma];
					}
					else {
						dkG_ijklk1_kk=dG_table[gamma][alpha-gamma-1][ijkl_ind][k][n1_alpha+n1_gamma];
					}
					//
					double tilde_W_i_s=0.;
					for (int t=0;t<i+1;++t) {
						double dx_odot_W=0.;
						int q_min=std::max(0,n1_gamma-t);
						int q_max=std::min(i-t,n1_gamma);
						for (int q=q_min;q<q_max+1;++q) {
							dx_odot_W+=Binom(i-t,q)*Binom(t,n1_gamma-q)\
							          *pow(dx_ga_al_1,q)\
							          *pow(dx_ga_al_2,i-t-q)\
							          *W(t+s,n1_gamma-q+ns1,t-(n1_gamma-q)+ns2);
						}
						dx_odot_W/=Binom(i,n1_gamma);
						//
						tilde_W_i_s+=Binom(i,t)*dx_odot_W;
					}
					//
					T_i+=1./(L*L)*pow(-1,i)/Factorial(k-i)/Factorial(i)\
								 *dkG_ijklk1_kk\
					             *fac_alpha*W(r+k-i,nr1+n1_alpha,nr2+n2_alpha)\
					             *fac_gamma*tilde_W_i_s;
				}
			}
		}
	}
	return T_i;
}