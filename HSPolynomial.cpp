#include <math.h>
#include <vector>
#include "GreenAnisotropic2D.hpp"

using namespace std;

vector<double> tau(int p, vector<double> tau0, vector<double> tau_grads, double dx1, double dx2) {
	// 


	double gn=pow(-r,-n)/M_PI;
	vector<double> nvec(2), mvec(2);
	nvec={cos(th),sin(th)};
	mvec={-sin(th),cos(th)};
	if (n>=1) {
		return gn*h(n,i,th)/2.;
	}	
	return 0;
}