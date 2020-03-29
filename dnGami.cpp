#include <math.h>
#include <vector>
#include "GreenAnisotropic2D.hpp"

double dnGami(medium mat, int k, std::vector<int> i, double r, double t) {
	double dnG=0.;
	//
	if (i[2]==i[3]) {
		if (i[0]==i[1]) {
			std::vector<int> ijkl={i[0],i[2],i[1],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			return dnG;
		}
		else {
			std::vector<int> ijkl={i[0],i[2],i[1],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			ijkl={i[1],i[2],i[0],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			return .5*dnG;
		}
	}
	else {
		if (i[0]==i[1]) {
			std::vector<int> ijkl={i[0],i[2],i[1],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			ijkl={i[0],i[3],i[1],i[2]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			return .5*dnG;
		}
		else {
			std::vector<int> ijkl={i[0],i[2],i[1],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			ijkl={i[0],i[3],i[1],i[2]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			ijkl={i[1],i[2],i[0],i[3]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			ijkl={i[1],i[3],i[0],i[2]};
			ijkl.insert(ijkl.end(),i.begin()+4,i.end());
			dnG+=mat.dnGi(k+2,ijkl,r,t);
			//
			return .25*dnG;
		}
	}
}