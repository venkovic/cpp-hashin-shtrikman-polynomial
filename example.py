import numpy as np
import scipy.linalg as lng
import pylab as pl
import matplotlib as mpl
from matplotlib import colors, ticker, cm
from scipy.integrate import quad
import scipy.special as spe
import scipy.optimize as opt

import GreenAnisotropic2D
import HSPolynomialpy as HSPolynomial
import itertools
import time



pl.rcParams['text.usetex'] = True
pl.rcParams['axes.labelsize'] = 17.
pl.rcParams['legend.fontsize']=18.
#pl.rcParams['legend.frameon'] = 'False'
pl.rcParams['legend.fontsize']=13.
pl.rcParams['xtick.labelsize']=13.
pl.rcParams['ytick.labelsize']=13.
pl.rcParams['legend.numpoints']=1

#fig_format='.eps'
fig_format='.png'

# Select symmetry
isym=0
if (isym==0):
	# Anisotropic case
	p=[.3,3.6,1.,.7,2.1,2.0]
if (isym==1):
	# Orthotrop+ic case 
	p=[.3,0.,.6,.8,1.2,2.5]
if (isym==2):
	# R0-Orthotropic case
	p=[.3,.5,1.2,2.7]
if (isym==3):
	# Square symmetric case
	p=[.3,.8,1.2,2.5]
if (isym==4):
	# Polar isotropic case
	p=[1.2,.3]
if (isym==5):
	# Isotropic case
	p=[1.2,.3]

#
# Define a medium object for the material
mat=GreenAnisotropic2D.medium()
#
# Set up symmetry
mat.set_sym(isym,p)

#
# Verify strain energy is positive and the symmetry is properly identified
status=HSPolynomial.check_pos(mat)
if (status==0):
	print "Strain energy is positive and symmetry is properly identified."
	#
	# Compute first complete Barnett-Lothe integral
	mat.get_S()
	# Compute second complete Barnett-Lothe integral
	mat.get_H()
elif (status==1):	
	print "Strain energy is negative."
elif (status==2):	
	print "Symmetry not properly identified."


###################################################################################################################
###################################################################################################################
###################################################################################################################
r=.4
t=.5
#
# Minor symmetry of 2nd gradient of Geen's function
print "\n Is the 2nd gradient of the Green's function minor symmetric? Yes,"
print mat.dnGi(2,[1,2,1,2],r,t)
print mat.dnGi(2,[1,2,2,1],r,t)
print mat.dnGi(2,[2,1,1,2],r,t)
print mat.dnGi(2,[2,1,2,1],r,t)

print "\n Is the 2nd gradient of the Green's function major symmetric? No,"
print mat.dnGi(2,[1,1,2,2],r,t)
print mat.dnGi(2,[2,2,1,1],r,t)
# This is a problem for slide 9
# ({}^nT^{a,b}_{0,0})_{ijkl} \neq ({}^nT^{b,a}_{0,0})_{klij}

print "\n Does the Green's function have full symmetry of derivatives? Yes,"
print mat.dnGi(2,[2,2,1,2],r,t)
print mat.dnGi(2,[2,2,2,1],r,t)
print mat.dnGi(5,[2,2,1,2,1,2,1],r,t)
print mat.dnGi(5,[2,2,1,1,1,2,2],r,t)


print "\n Symmetry in x?"
print "d^0Gi: ", mat.dnGi(0,[1,2],r,t), mat.dnGi(0,[1,2],r,t+np.pi)
print "d^1Gi: ", mat.dnGi(1,[1,2,1],r,t), mat.dnGi(1,[1,2,1],r,t+np.pi)
print "d^2Gi: ", mat.dnGi(2,[1,2,1,2],r,t), mat.dnGi(2,[1,2,1,2],r,t+np.pi)
print "d^3Gi: ", mat.dnGi(3,[1,2,1,2,1],r,t), mat.dnGi(3,[1,2,1,2,1],r,t+np.pi)
print "d^4Gi: ", mat.dnGi(4,[1,2,1,2,1,2],r,t), mat.dnGi(4,[1,2,1,2,1,2],r,t+np.pi)
print "d^5Gi: ", mat.dnGi(5,[1,2,1,2,1,2,1],r,t), mat.dnGi(5,[1,2,1,2,1,2,1],r,t+np.pi)
print "d^6Gi: ", mat.dnGi(6,[1,2,1,2,1,2,1,2],r,t), mat.dnGi(6,[1,2,1,2,1,2,1,2],r,t+np.pi)
print "d^7Gi: ", mat.dnGi(7,[1,2,1,2,1,2,1,2,1],r,t), mat.dnGi(7,[1,2,1,2,1,2,1,2,1],r,t+np.pi)
print "d^8Gi: ", mat.dnGi(8,[1,2,1,2,1,2,1,2,1,2],r,t), mat.dnGi(8,[1,2,1,2,1,2,1,2,1,2],r,t+np.pi)

#
# Gradient of Green operator for strain
def dnGami_old(mat,n,i,r,t):
	return .25*(\
	mat.dnGi(n+2,[i[0],i[2],i[1],i[3]]+i[4:],r,t)+\
	mat.dnGi(n+2,[i[0],i[3],i[1],i[2]]+i[4:],r,t)+\
	mat.dnGi(n+2,[i[1],i[2],i[0],i[3]]+i[4:],r,t)+\
	mat.dnGi(n+2,[i[1],i[3],i[0],i[2]]+i[4:],r,t))
#
# n-th order gradient of Green operator for strain // taking advantage of symmetries
# len(i)=n+2  
def dnGami(mat,n,i,r,t):
	if (i[2]==i[3]):
		if (i[0]==i[1]):
			return mat.dnGi(n+2,[i[0],i[2],i[1],i[3]]+i[4:],r,t)
		else:
			return .5*(\
			mat.dnGi(n+2,[i[0],i[2],i[1],i[3]]+i[4:],r,t)+\
			mat.dnGi(n+2,[i[1],i[2],i[0],i[3]]+i[4:],r,t))
	else:
		if (i[0]==i[1]):
			return .5*(\
			mat.dnGi(n+2,[i[0],i[2],i[1],i[3]]+i[4:],r,t)+\
			mat.dnGi(n+2,[i[0],i[3],i[1],i[2]]+i[4:],r,t))
		else:
			return .25*(\
			mat.dnGi(n+2,[i[0],i[2],i[1],i[3]]+i[4:],r,t)+\
			mat.dnGi(n+2,[i[0],i[3],i[1],i[2]]+i[4:],r,t)+\
			mat.dnGi(n+2,[i[1],i[2],i[0],i[3]]+i[4:],r,t)+\
			mat.dnGi(n+2,[i[1],i[3],i[0],i[2]]+i[4:],r,t))
		



import numpy as np
import scipy.linalg as lng
import pylab as pl
import GreenAnisotropic2D
isym=0
p=[.3,3.6,1.,.7,2.1,2.0]
mat=GreenAnisotropic2D.medium()
mat.set_sym(isym,p)
mat.get_S()
mat.get_H()
r=2.
t=.4
dx=.0000000000001
xb=r*np.cos(t)+dx
xa=r*np.cos(t)-dx
ya=r*np.sin(t)
yb=r*np.sin(t)
ra=np.sqrt(xa**2+ya**2)
ta=np.arctan2(ya,xa)
rb=np.sqrt(xb**2+yb**2)
tb=np.arctan2(yb,xb)
print '\n'
print (mat.dnGi(1,[1,1,2],rb,tb)-mat.dnGi(1,[1,1,2],ra,ta))/2./dx
print mat.dnGi(2,[1,1,2,1],r,t)
print '\n'
print (mat.dnGi(2,[1,1,2,1],rb,tb)-mat.dnGi(2,[1,1,2,1],ra,ta))/2./dx
print mat.dnGi(3,[1,1,2,1,1],r,t)
print '\n'
print (mat.dnGi(3,[1,1,2,1,2],rb,tb)-mat.dnGi(3,[1,1,2,1,2],ra,ta))/2./dx
print mat.dnGi(4,[1,1,2,1,2,1],r,t)
print '\n'
print (mat.dnGi(4,[1,1,2,1,2,1],rb,tb)-mat.dnGi(4,[1,1,2,1,2,1],ra,ta))/2./dx
print mat.dnGi(5,[1,1,2,1,2,1,1],r,t)
print '\n'
print (mat.dnGi(5,[1,1,2,1,2,1,2],rb,tb)-mat.dnGi(5,[1,1,2,1,2,1,2],ra,ta))/2./dx
print mat.dnGi(6,[1,1,2,1,2,1,2,1],r,t)
print '\n'
print (mat.dnGi(6,[1,1,2,1,2,1,2,1],rb,tb)-mat.dnGi(6,[1,1,2,1,2,1,2,1],ra,ta))/2./dx
print mat.dnGi(7,[1,1,2,1,2,1,2,1,1],r,t)
print '\n'
print (mat.dnGi(6,[1,2,1,2,1,2,1,2],rb,tb)-mat.dnGi(6,[1,2,1,2,1,2,1,2],ra,ta))/2./dx
print mat.dnGi(7,[1,2,1,2,1,2,1,2,1],r,t)






#
r=.4
t=.5
print "\n Is the Green operator for strain minor symmetric? Yes,"
print dnGami(mat,0,[1,2,1,2],r,t)
print dnGami(mat,0,[1,2,2,1],r,t)
print dnGami(mat,0,[2,1,1,2],r,t)
print dnGami(mat,0,[2,1,2,1],r,t)


print "\n Is the Green operator for strain major symmetric? Yes,"
print dnGami(mat,0,[1,1,1,2],r,t)
print dnGami(mat,0,[1,2,1,1],r,t)



# d^(k)N1 and d^(k)N2 are implemented for k=0 ... 7.
# -> Allows computations for n=1 ... 6, irrespectively of p
#
# (p,n)
# (2,2), (2,3), (2,4), (2,5), (2,6)
# (3,3), (3,4), (3,5), (3,6)
# (4,4), (4,5), (4,6)
# (5,5), (5,6)
#

#
# Start with the assumption that we should have n>=p
#
#
# Order of polynomial polarization field
p=6
#
# Order of the Taylor expansion
n=6

# Derine heterogeneous medium made of phases
n_al=4
mat=[]
#params=[[.3,3.6,1.,.7,1.1,2.0],[.7,1.6,.4,.9,1.2,2.4],[.4,1.7,1.,.5,1.4,4.2],[.3,1.6,1.,.8,1.2,2.4]]; isym=0
params=[[.1,1.7,1.,.5,1.4,4.2],[.4,1.7,1.,.5,1.4,4.2],[.6,1.7,1.,.5,1.4,4.2],[1.4,1.7,1.,.5,1.4,4.2]]; isym=0
#params=[[1.2,.3],[.7,.8],[2.,.9],[1.7,.6]]; isym=5

for al in range(n_al):
	mat.append(HSPolynomial.medium2D(isym))
	mat[-1].set_stiff_polar(params[al])

# Upper bound
if (isym==0):
	cte_PSD_Voigt=2.5407 # Lower factor by which we multiply the Voigt stiffness estimate for SPSD
else:
	cte_PSD_Voigt=1.45
# Eventually, need to find a way to estimate the bounding stiffness
L0_Voigt_mandel=np.zeros((3,3))
for k in range(n_al):
	L0_Voigt_mandel+=mat[k].L_mandel
L0_Voigt_mandel/=n_al
L0_Voigt_mandel*=cte_PSD_Voigt
print "\n"
# Verify that (Voigt stiffness estimate - local stiffness) is PSD everywhere 
for k in range(n_al):
	print "Eigvals for L0_Voigt-Li:",np.linalg.eigvals(L0_Voigt_mandel-mat[k].L_mandel)


# Lower bound
if (isym==0):
	cte_PSD_Reuss=.36126 # Highest factor by which we multiply the Reuss stiffness estimate for SPSD
else:
	cte_PSD_Reuss=.55
# Eventually, need to find a way to estimate the bounding stiffness
M0_Reuss_mandel=np.zeros((3,3))
for k in range(n_al):
	M0_Reuss_mandel+=mat[k].M_mandel
M0_Reuss_mandel/=n_al
L0_Reuss_mandel=np.linalg.inv(M0_Reuss_mandel)
L0_Reuss_mandel*=cte_PSD_Reuss
print "\n"
# Verify that (local stiffness - Reuss stiffness estimate) is PSD everywhere 
for k in range(n_al):
	print "Eigvals for Li-L0_Reuss:",np.linalg.eigvals(mat[k].L_mandel-L0_Reuss_mandel)


# Define reference stiffness with desired SPSD property
# Need to find corresponding polar anisotropic formulation
L0=L0_Reuss_mandel
if (isym==0):
	T0=(L0[0][0]-2.*L0[0][1]+4.*L0[2][2]/2.+L0[1][1])/8.
	T1=(L0[0][0]+2*L0[0][1]+L0[1][1])/8.
	R0=np.sqrt((L0[0][0]-2.*L0[0][1]-4.*L0[2][2]/2.+L0[1][1])**2+16*(L0[0][2]/np.sqrt(2)-L0[1][2]/np.sqrt(2.))**2)/8.
	R1=np.sqrt((L0[0][0]-L0[1][1])**2+4.*(L0[0][2]/np.sqrt(2)+L0[1][2]/np.sqrt(2))**2)/8.
	Phi0=np.arctan2(4.*(L0[0][2]/np.sqrt(2)-L0[1][2]/np.sqrt(2.)),L0[0][0]-2.*L0[0][1]-4.*L0[2][2]/2.+L0[1][1])/4.
	Phi1=np.arctan2(2.*(L0[0][2]/np.sqrt(2)+L0[1][2]/np.sqrt(2)),L0[0][0]-L0[1][1])/2.
	p0=[Phi0,Phi1,R0,R1,T0,T1]
else:
	T0=(L0[0][0]-2.*L0[0][1]+4.*L0[2][2]/2.+L0[1][1])/8.
	T1=(L0[0][0]+2*L0[0][1]+L0[1][1])/8.
	kappa=(L0[0][0]+L0[0][1])/2.
	mu=L0[2][2]/2.
	p0=[kappa,mu]
mat0_Green=GreenAnisotropic2D.medium()
mat0_Green.set_sym(isym,p0)
mat0_Green.get_H()
mat0_Green.get_S()
mat0=HSPolynomial.medium2D(isym)
mat0.set_stiff_polar(p0)






"""

##############################################
##############################################


mat_opt=HSPolynomial.medium2D(isym=0)

def func(p, sign=1.):
	return np.sum(dL_eigvals(p)**2)/100.

def dL_eigvals(p, sign=1.):
	mat_opt.set_stiff_polar(p)
	Lopt_mandel=mat_opt.L_mandel
	return -np.concatenate([np.linalg.eigvals(Lopt_mandel-mat[al].L_mandel) for al in range(n_al)])

tol=10.**-6
def pos_def_energy(p):
	return np.array([p[4]-p[2],\
		             p[5]*(p[4]**2-p[2]**2)-2.*p[3]**2*(p[4]-p[2]*np.cos(4.*(p[0]-p[1]))),\
		             p[2],\
		             p[3]])-tol
def cons_func(p,sign=1.):
	return np.concatenate([dL_eigvals(p),pos_def_energy(p)])

cons = ({'type': 'ineq',
         'fun' : lambda x: cons_func(x)})

#res = opt.minimize(func, p0, args=(1.0,), constraints=cons, method='SLSQP', options={'disp': True, 'maxiter': 10000})
res = opt.minimize(func, params[0], args=(1.0,), constraints=cons, options={'disp': True, 'maxiter': 100000})

print p0, dL_eigvals(p0)
print res.x, dL_eigvals(res.x)
print ""





def DE_func(p):
	if np.any(cons_func(p)<0):
		return 10000.*np.random.rand()
	else:
		return func(p)


bounds = [(0, 2.*np.pi), (0, 2.*np.pi), (0, 10), (0, 10), (0, 10), (0, 10)]
res=opt.differential_evolution(DE_func, bounds)
#result.x, result.fun
#(array([ 0.,  0.]), 4.4408920985006262e-16)



##############################################
##############################################

"""























print mat0_Green.S
print mat0_Green.H


# mandel_to_mat(A_mandel)
#	-> Transform the Mandel representation of a 4th order tensor into a simple array of components
#
# W(r,ni1,ni2)
#	-> Compute a component of W^{r,0}_0 for a square centered at the origin									// Verified (slide 61)
#
# min_diff(x_al,x_be,L)																						// Verified	
#	-> Find the minimum distance between the center of the squares {al} and {be} in a periodic array
#
# T_global(n,s,r,n_al)																						// Verified (slide 34)
#	-> Compute global Mandel matrix of estimates of r-s influence tensors
#
# T_sym_infl(n,al,be,nr1,nr2,ijkl,ns1,ns2,dx_r,dx_th)														// Verified (slide 14)
#	-> Compute specific component of estimate of r-s influence tensor of {be} over {al} 
#
# T_local(n,al,be,s,r)
#	-> Compute Mandel matrix of estimate of r-s influence tensor of {be} over {al} 							// Verified (slide 28-33)
#
# M_global(s,r,n_al)																						// Verified (slide 27)									
#	-> Compute global Mandel matrix of r-s Minkowski-weighted tensors
#
# dMW_local(al,s,r)
#	-> Compute Mandel matrix of r-s Minkowski-weighted tensor of {al}										// Verified (slide 26-27)
#
# D_global(n,s,r,n_al)																						// Verified (slide 34) 
#	-> Compute global r-s compliance of the system
#
# D_mat_assemble(p,n,n_al)																					// Verified (slide 35)
#	-> Compute compliance of the overall system
#
# eps_global0(n_al,eps_av_mandel)																			// Verified (slide 25)
#	-> Compute Mandel vector of 0th Minkowski-weigthed average strain
#
# eps_global(p,n_al,eps_av_mandel)																			// Verified (slide 25)	
#	-> Compute Mandel vector of higher orders Minkowski-weigthed average strain	
#
# tau(p,tau0,tau_grads,al,dx1,dx2)																			// Verified (slide 35)
#	-> Compute polarization stress state at a given point within the plane
#






# Return ordinary matrix representation of a Mandel representation
def mandel_to_mat(A_mandel):
	A=np.array([[A_mandel[0][0],A_mandel[0][1],A_mandel[0][2]/np.sqrt(2)],
				[A_mandel[1][0],A_mandel[1][1],A_mandel[1][2]/np.sqrt(2)],
				[A_mandel[2][0]/np.sqrt(2),A_mandel[2][1]/np.sqrt(2),A_mandel[2][2]/2.]])
	return A

#
# Compute compliance differences
dM_mandel=[]
LdM_mandel=[]
dM=[]
LdM=[]
for k in range(n_al):
	dM_mandel.append(np.linalg.inv(mat[k].L_mandel-mat0.L_mandel))
	LdM_mandel.append(np.dot(mat[k].L_mandel,dM_mandel[k]))
	dM.append(mandel_to_mat(dM_mandel[k]))
	LdM.append(mandel_to_mat(LdM_mandel[k]))


#
# Return components of translated Minkowski tensors
def W(r,ni1,ni2):
	# ni1+ni2=r
	a=L/2.
	Wr_i=((a/2.)**(ni1+ni2+2)\
		 -(-a/2.)**(ni1+1)*(a/2.)**(ni2+1)\
		 -(a/2.)**(ni1+1)*(-a/2.)**(ni2+1)\
		 +(-a/2.)**(ni1+1)*(-a/2.)**(ni2+1))/(ni1+1)/(ni2+1)
	return Wr_i
#	
# Set up volume fractions
cfrac=np.array(n_al*[1./n_al])
#
# Set up centers of gravity
L=10.
xloc=[np.array((2.5,2.5)),np.array((7.5,2.5)),np.array((7.5,7.5)),np.array((2.5,7.5))]
#
# Define vector between two points of periodic medium with minimum magnitude
def min_diff(x_gamma,x_alpha,L):
	dx=x_alpha-x_gamma
	if (dx[0]>L/2.):
		dx[0]=-(L-dx[0])
	elif (dx[0]<-L/2.):
		dx[0]=L+dx[0]
	if (dx[1]>L/2.):
		dx[1]=-(L-dx[1])
	elif (dx[1]<-L/2.):
		dx[1]=L+dx[1]
	return dx

#
# Return global influence matrix
def T_global(n,s,r,n_al):
	# Makes calls to T_local()
	T=np.zeros((3*(r+1)*n_al,3*(s+1)*n_al))

#	print "n,s,r,n_al=",n,s,r,n_al
	for al in range(n_al):
		ddim_i=3*(r+1)
		ddim_j=3*(s+1)
		cum_T_loc=np.zeros((ddim_i,ddim_j))
		for be in range(n_al):
			if (al!=be):
				#T_loc=T_local(n,al,be,s,r)

				t_startT=time.time()
				T_loc=T_local(n,be,al,s,r)
				t_endT=time.time()
				print "   T_local(", n,be,al,s,r, ") dt=", t_endT-t_startT				
				cum_T_loc+=T_loc
				#T[ddim_i*be:ddim_i*(be+1),ddim_j*al:ddim_j*(al+1)]=T_loc
				T[ddim_i*al:ddim_i*(al+1),ddim_j*be:ddim_j*(be+1)]=T_loc
		T[ddim_i*al:ddim_i*(al+1),ddim_j*al:ddim_j*(al+1)]=-cum_T_loc

	return T


def counter(list_of_els,el):
	ni=0
	for eli in list_of_els:
		if (eli==el):
			ni+=1
	return ni


def T_sym_infl(n,al,be,nr1,nr2,ijkl_ind,ns1,ns2,dx_r,dx_th):
	r=nr1+nr2
	s=ns1+ns2
#	print "0,ijkl,dx_r,dx_th,",0,ijkl,dx_r,dx_th
	#T_i=cfrac[al]*cfrac[be]*W(r,nr1,nr2)*dnGami(mat0_Green,0,ijkl,dx_r,dx_th)*W(s,ns1,ns2)
	if (be>al):
		#print ijkl_ind
		#print dG_table[al][be-al-1][ijkl_ind][0][0]
		dkG_ijkl=dG_table[al][be-al-1][ijkl_ind][0][0]
	else:
		dkG_ijkl=dG_table[be][al-be-1][ijkl_ind][0][0]
	T_i=W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2)/L**2
#	T_i=cfrac[al]*cfrac[be]*W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2)
#	T_ii=cfrac[al]*cfrac[be]*W(r,nr1,nr2)*dkG_ijkl*W(s,ns1,ns2)
#	print T_i
	for k in range(1,n+1):
		#
		for i in range(k+1):

			for i_alpha in range(k-i+1):
				if not (i_alpha%2): 							# 0, 2,   4,   6,   ..., i
					n1_alpha=(k-i)-i_alpha/2   					# i, i-1, i-2, i-3, ..., (i+1)/2
					n2_alpha=(k-i)-n1_alpha						# 0, 1,   2,   3,   ..., i-(i+1)/2
				#
				elif (i_alpha%2):								# 1, 3,   5,   7,   ..., i
					n2_alpha=(k-i)-(i_alpha-1)/2   				# i, i-1, i-2, i-3, ..., (i+1)/2
					n1_alpha=(k-i)-n2_alpha 						# 0, 1,   2,   3,   ..., i-(i+1)/2
				#
				fac_ii=spe.binom(k-i,n1_alpha)
				#
				for i_gamma in range(i+1):
					if not (i_gamma%2): 						# 0, 2,   4,   6,   ..., (k-i)
						n1_gamma=i-i_gamma/2   					# i, i-1, i-2, i-3, ..., ((k-i)+1)/2
						n2_gamma=i-n1_gamma						# 0, 1,   2,   3,   ..., (k-i)-((k-i)+1)/2
					#
					elif (i_gamma%2):							# 1, 3,   5,   7,   ..., (k-i)
						n2_gamma=i-(i_gamma-1)/2   				# i, i-1, i-2, i-3, ..., ((k-i)+1)/2
						n1_gamma=i-n2_gamma 					# 0, 1,   2,   3,   ..., (k-i)-((k-i)+1)/2
					#
					fac_jj=spe.binom(i,n1_gamma)
					#
					#dkG_ijklk1_kk=dnGami(mat0_Green,k,ijkl+\
					#	(n1_from_k_1_to_k_kminusi+n1_from_k_kminusip1_to_k_k)*[1]+\
					#	(n2_from_k_1_to_k_kminusi+n2_from_k_kminusip1_to_k_k)*[2],dx_r,dx_th)
					#print k, ijkl, n1_from_k_1_to_k_kminusi+n1_from_k_kminusip1_to_k_k, n2_from_k_1_to_k_kminusi+n2_from_k_kminusip1_to_k_k
					if (be>al):
						#print len(dG_table),len(dG_table[al]),len(dG_table[al][be-al-1]),len(dG_table[al][be-al-1][list_of_ijkl.index(ijkl)]),len(dG_table[al][be-al-1][list_of_ijkl.index(ijkl)][k])
						#print al,be-al-1,list_of_ijkl.index(ijkl),k,n1_from_k_1_to_k_kminusi+n1_from_k_kminusip1_to_k_k
						dkG_ijklk1_kk=(-1)**k*dG_table[al][be-al-1][ijkl_ind][k][n1_alpha+n1_gamma]
					else:
						dkG_ijklk1_kk=dG_table[be][al-be-1][ijkl_ind][k][n1_alpha+n1_gamma]
					#print dkG_ijklk1_kk, dkG_ijklk1_kk_bis
					
#					T_i+=(L**2)**-1*(-1)**i/spe.factorial(k-i)/spe.factorial(i)/2.*dkG_ijklk1_kk\
#					          *(W(r+k-i,nr1+n1_from_k_1_to_k_kminusi,nr2+n2_from_k_1_to_k_kminusi)\
#					           *W(s+i,ns1+n1_from_k_kminusip1_to_k_k,ns2+n2_from_k_kminusip1_to_k_k)\
#					   +(-1)**k*W(r+i,nr1+n1_from_k_kminusip1_to_k_k,nr2+n2_from_k_kminusip1_to_k_k)\
#					           *W(s+k-i,ns1+n1_from_k_1_to_k_kminusi,ns2+n2_from_k_1_to_k_kminusi))*fac_ii*fac_jj

					#T_i+=(L**2)**-1*(-1)**i/spe.factorial(k-i)/spe.factorial(i)/2.*dkG_ijklk1_kk\
					#          *(W(r+k-i,nr1+n1_from_k_1_to_k_kminusi,nr2+n2_from_k_1_to_k_kminusi)\
					#           *W(s+i,ns1+n1_from_k_kminusip1_to_k_k,ns2+n2_from_k_kminusip1_to_k_k)\
					#           +W(r+i,nr1+n1_from_k_kminusip1_to_k_k,nr2+n2_from_k_kminusip1_to_k_k)\
					#           *W(s+k-i,ns1+n1_from_k_1_to_k_kminusi,ns2+n2_from_k_1_to_k_kminusi))*fac_ii*fac_jj

					T_i+=(L**2)**-1*(-1)**i/spe.factorial(k-i)/spe.factorial(i)*dkG_ijklk1_kk\
					          *(W(r+k-i,nr1+n1_alpha,nr2+n2_alpha)\
					           *W(s+i,ns1+n1_gamma,ns2+n2_gamma))*fac_ii*fac_jj				          
#	print T_i, T_ii
	return T_i

# To check:
#	- Maxwell-Betti -> (T^{al,be}_{0,0})_{ijkl} = (T^{be,al}_{0,0})_{klij}		// See Brisard (2010), Eq. (1.22b)
#	- Maxwell-Betti higher orders?												// See Brisard (2010), Eq. ()
#
#	See (D.3a-c) in Brisard (2010).
#		- A_{k,al be} = (T^{al,be}_{1,0})_k
#		- A^\prime_{l,al be} = (T^{al,be}_{0,1})_l
#		- A_{kl,al be} = (T^{al,be}_{1,1})_{kl}
#
#	0/ Do a last check-up of the generalized Mandel notation.
#
#   I/ Then. Do an upper triangular implementation.
#
#	II/ C++ implementation.
#
#	III/a Compute error in equilibrium.
#	III/b Make an implementation with local equiblibrium enforced?
#
#	IV/ Condition used to compute self-influence tensors. Relation with BCs?
#
#   V/ Get a sense of orders at play from Brisard's implementation.
#
#	VI/ Why are shear estimates behaving so crazily?
# 
# Return local influence matrix // Follow slides 11 and 20-25

list_of_ijkl=[[1,1,1,1],[1,1,2,2],[1,1,1,2],\
						[2,2,2,2],[2,2,1,2],\
								  [1,2,1,2]];
		
list_of_ijkl_inds=[[0,1,2],\
				   [1,3,4],\
				   [2,4,5]]	

def T_local(n,be,al,s,r):
	# Call made: T_local(n,be,al,s,r)
	#
	T=np.zeros((3*(r+1),3*(s+1)))
	#
	#
	# Compute (r,t) given by minimum difference between centers of gravity of {al} and {be}
	dx_al_be=min_diff(xloc[al],xloc[be],L)
	dx_r=np.sqrt(dx_al_be[0]**2+dx_al_be[1]**2)
	dx_th=np.arctan2(dx_al_be[1],dx_al_be[0])
	#
	ij=[[1,1],[2,2],[1,2]]
	fac_ij=[1.,1.,np.sqrt(2)]
	#
	for i in range(r+1):
		if not (i%2): 		# i   = 0, 2,   4,   6,   ..., r
			ni1=r-i/2   	# ni1 = r, r-1, r-2, r-3, ..., (r+1)/2
			ni2=r-ni1		# ni2 = 0, 1,   2,   3,   ..., r-(r+1)/2
		#
		elif (i%2):			# i   = 1, 3,   5,   7,   ..., r
			ni2=r-(i-1)/2   # ni2 = r, r-1, r-2, r-3, ..., (r+1)/2
			ni1=r-ni2       # ni1 = 0, 1,   2,   3,   ..., r-(r+1)/2
		#
		# Get number of indices combinations used for Mandel representation
		fac_i=np.sqrt(spe.binom(r,ni1))
		for k in range(len(ij)):
			for j in range(s+1):
				if not (j%2): 		# j   = 0, 2,   4,   6,   ..., s
					nj1=s-j/2   	# nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
					nj2=s-nj1		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
				#
				elif (j%2): 		# j   = 0, 2,   4,   6,   ..., s
					nj2=s-(j-1)/2   # nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
					nj1=s-nj2		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
				#
				# Get number of indices combinations used for Mandel representation
				fac_j=np.sqrt(spe.binom(s,nj1))
				#
				#for kk in range(len(ij)):
				for kk in range(len(ij)):
					#ijkl=ij[k]+ij[kk]
					#T[3*i+k][3*j+kk]=fac_i*fac_ij[k]*fac_ij[kk]*fac_j*T_sym_infl(n,al,be,ni1,ni2,ijkl,nj1,nj2,dx_r,dx_th)	Not consistent with slides. Does not matter because of symmetry
					T[3*i+k][3*j+kk]=fac_i*fac_ij[k]*fac_ij[kk]*fac_j*T_sym_infl(n,be,al,nj1,nj2,list_of_ijkl_inds[k][kk],ni1,ni2,dx_r,dx_th)
					#T[3*i+kk][3*j+k]=T[3*i+k][3*j+kk]
	return T

#
# Return global Minkowski-weighted compliance matrix (generalized Mandel representation) // Follow slide 19
def M_global(s,r,n_al):
	# Pay attention: dM is a global object
	dMW=np.zeros((3*(r+1)*n_al,3*(s+1)*n_al))
	ddim_i=3*(r+1)
	ddim_j=3*(s+1)
	if (r==0)&(s==0):
		for k in range(n_al):
			#dMW[ddim_i*k:ddim_i*(k+1),ddim_j*k:ddim_j*(k+1)]=cfrac[k]*dM[k]
			dMW[ddim_i*k:ddim_i*(k+1),ddim_j*k:ddim_j*(k+1)]=cfrac[k]*dM_mandel[k]
	else:
		for k in range(n_al):
			dMW[ddim_i*k:ddim_i*(k+1),ddim_j*k:ddim_j*(k+1)]=dMW_local(k,s,r)
	return dMW

#
# Return local Minkowski-weighted compliance matrix (generalized Mandel representation) // Follow slide 19
def dMW_local(al,s,r):
	# Pay attention: dM is a global object
	dMW=np.zeros((3*(r+1),3*(s+1)))
	for i in range(r+1):
		if not (i%2): 		# i   = 0, 2,   4,   6,   ..., r
			ni1=r-i/2   	# ni1 = r, r-1, r-2, r-3, ..., (r+1)/2
			ni2=r-ni1		# ni2 = 0, 1,   2,   3,   ..., r-(r+1)/2
		#
		elif (i%2):			# i   = 1, 3,   5,   7,   ..., r
			ni2=r-(i-1)/2   # ni2 = r, r-1, r-2, r-3, ..., (r+1)/2
			ni1=r-ni2       # ni1 = 0, 1,   2,   3,   ..., r-(r+1)/2
		#
		# Get number of indices combinations used for Mandel representation
		fac_i=np.sqrt(spe.binom(r,ni1))
		for j in range(s+1):
			if not (j%2): 		# j   = 0, 2,   4,   6,   ..., s
				nj1=s-j/2   	# nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
				nj2=s-nj1		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
			#
			elif (j%2): 		# j   = 0, 2,   4,   6,   ..., s
				nj2=s-(j-1)/2   # nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
				nj1=s-nj2		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
			#
			# Get number of indices combinations used for Mandel representation
			fac_j=np.sqrt(spe.binom(s,nj1))
			#
			# Populate local Mandel matrix of Minkowski-weigthed compliance differential
			dMW[3*i:3*(i+1),3*j:3*(j+1)]=fac_i*fac_j*W(r+s,ni1+nj1,ni2+nj2)*dM_mandel[al]
	return dMW

#
# Return global D_s^r matrix
def D_global(n,s,r,n_al):
	# size: 3*(r+1)*n_al x 3*(s+1)*n_al
	t_startM=time.time()
	M_=M_global(s,r,n_al)
	t_endM=time.time()
	t_startT=time.time()
	T_=T_global(n,s,r,n_al)
	t_endT=time.time()
	print "M", np.shape(M_), "dt=", t_endM-t_startM, " // T", np.shape(T_), "dt=", t_endT-t_startT
	return M_+T_

#
# Assemble D matrix 
def D_mat_assemble(p,n,n_al):
	Dmat=np.zeros((n_al*3*sum(range(2,p+2)),n_al*3*sum(range(2,p+2))))
	Dmat_sizes=[0]+[3*n_al*(i+1) for i in range(1,p+1)]
	for r in range(1,p+1):
		for s in range(1,p+1):
			i_start=sum(Dmat_sizes[:r])
			i_end=sum(Dmat_sizes[:r+1])
			j_start=sum(Dmat_sizes[:s])
			j_end=sum(Dmat_sizes[:s+1])				
			print r, s, i_start, i_end, j_start, j_end, np.shape(Dmat[i_start:i_end,j_start:j_end])
			Dmat[i_start:i_end,j_start:j_end]+=D_global(n,s,r,n_al)

	return Dmat
#
# Assemble average volume fraction-weighted strain vector
def eps_global0(n_al,eps_av_mandel):
	eps=np.zeros(3*n_al)
	for al in range(n_al):
		eps[3*al:3*(al+1)]=np.array(cfrac[al]*eps_av_mandel)
	return eps

#
# Assemble average Minkowski-weighted strain vector
def eps_global(p,n_al,eps_av_mandel):
	eps=np.zeros(n_al*3*sum(range(2,p+2)))
	eps_sizes=[0]+[3*n_al*(i+1) for i in range(1,p+1)]
	for n_i in range(1,p+1):
		for al in range(n_al):
			for i in range(n_i+1):
				if not (i%2): 			# i   = 0,   2,     4,     6,     ..., n_i
					ni1=n_i-i/2 		# ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
					ni2=n_i-ni1			# ni2 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
				#
				elif (i%2):				# i   = 1,   3,     5,     7,     ..., n_i
					ni2=n_i-(i-1)/2 	# ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
					ni1=n_i-ni2 		# ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
				#
				# Get number of indices combinations used for Mandel representation
				fac_i=np.sqrt(spe.binom(n_i,ni1))
				#
				i_start=sum(eps_sizes[:n_i]) 			
				eps[i_start+al*(n_i+1)*3+i*3:i_start+al*(n_i+1)*3+(i+1)*3]=fac_i*W(n_i,ni1,ni2)*eps_av_mandel
	return eps

#
# Compute polarization stress state at a given location
def tau(p,tau0,tau_grads,al,dx1,dx2):
	#
	# Get in-grain constant part of the polarization stress field
	tau=np.copy(tau0[3*al:3*(al+1)])
	#
	tau_sizes=[0]+[3*n_al*((k-1)**2+3*(k-1))/2 for k in range(1,p+1)]
	#
	for k in range(1,p+1):
		#
		i_start=tau_sizes[k]
		for i in range(k+1):
			if not (i%2): 			# i   = 0,   2,     4,     6,     ..., n_i.
				ni1=k-i/2 		# ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni2=k-ni1			# ni2 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			elif (i%2):				# i   = 1,   3,     5,     7,     ..., n_i
				ni2=k-(i-1)/2 	# ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni1=k-ni2 		# ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			# Get number of indices combinations used for Mandel representation
			fac_i=np.sqrt(spe.binom(k,ni1))
			#tau+=fac_i*tau_grads[3*al*(n_i+1)+3*i:3*al*(n_i+1)+3*(i+1)]*dx1**ni1*dx2**ni2
			tau+=fac_i*tau_grads[i_start+al*(k+1)*3+i*3:i_start+al*(k+1)*3+(i+1)*3]*dx1**ni1*dx2**ni2
			#tau+=tau_grads[i_start+al*(n_i+1)*3+i*3:i_start+al*(n_i+1)*3+(i+1)*3]*dx1**ni1*dx2**ni2
	tau[2]/=np.sqrt(2)
	return tau




def error_div(p,tau_grads,al,dx1,dx2):
	#
	# Initilize divergence
	div_tau=np.zeros(2)
	#
	tau_sizes=[0]+[3*n_al*(i+1) for i in range(1,p+1)]
	#
	for k in range(1,p+1):
		for i in range(k):
			if not (i%2): 			# i   = 0,   2,     4,     6,     ..., n_i.
				ni1=k-i/2 		# ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni2=k-ni1			# ni2 = 0   ,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			elif (i%2):				# i   = 1,   3,     5,     7,     ..., n_i
				ni2=k-(i-1)/2 	# ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni1=k-ni2 		# ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			# Get number of indices combinations used for Mandel representation
			print "1:", k, ni1
			print "2:", k-1,ni2
			print "3:", k-1,ni1                                        

			fac_den=np.sqrt(spe.binom(k,ni1))
			#
			i_start=sum(tau_sizes[:k])
			if (ni1>0):
				fac1=spe.binom(k-1,ni2)/fac_den
				div_tau[0]+=k*fac1*(tau_grads[i_start+al*(k+1)*3+i*3+0]+tau_grads[i_start+al*(k+1)*3+i*3+2]/np.sqrt(2))*dx1**(ni1-1)*dx2**ni2
			if (ni2>0):
				fac2=spe.binom(k-1,ni1)/fac_den
				div_tau[1]+=k*fac2*(tau_grads[i_start+al*(k+1)*3+i*3+2]/np.sqrt(2)+tau_grads[i_start+al*(k+1)*3+i*3+1])*dx1**ni1*dx2**(ni2-1)
	return np.sqrt(div_tau[0]**2+div_tau[1]**2)






dG_table=[]
list_of_ijkl=[[1,1,1,1],[1,1,2,2],[1,1,1,2],\
						[2,2,2,2],[2,2,1,2],\
								  [1,2,1,2]]


print "\n Computing table of derivatives of the Green operator"
t_start=time.time()
for al in range(n_al-1):
	dG_table.append([])
	for be in range(al+1,n_al):
		#dx_al_be=min_diff(xloc[al],xloc[be],L)
		dx_al_be=min_diff(xloc[be],xloc[al],L)
		dx_r=np.sqrt(dx_al_be[0]**2+dx_al_be[1]**2)
		dx_th=np.arctan2(dx_al_be[1],dx_al_be[0])
		dG_table[al].append([])
		print dx_r, dx_th
		for i_ijkl in range(len(list_of_ijkl)):
			dG_table[al][be-al-1].append([])
			for k in range(n+1):
				dG_table[al][be-al-1][i_ijkl].append([])
				for i1 in range(k+1):
					dG_table[al][be-al-1][i_ijkl][k].append(dnGami(mat0_Green,k,list_of_ijkl[i_ijkl]+i1*[1]+(k-i1)*[2],dx_r,dx_th))





#print T_sym_infl(5,0,2,3,2,2,3,1,.4,.5)

print mat0_Green.S
print dG_table[1][1][1][1][1]

t_end=time.time()
print "dt = ", t_end-t_start

t_start=time.time()
eps_av_mandel=np.array([0.,1.,0.])
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing eps0"
t_start=time.time()
eps0=eps_global0(n_al,eps_av_mandel)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing eps_grads"
t_start=time.time()
eps_grads=eps_global(p,n_al,eps_av_mandel)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing Dmat0"
t_start=time.time()
Dmat0=D_global(p,0,0,n_al)
Dmat0_inv=np.linalg.inv(Dmat0)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing Dmat"
t_start=time.time()
Dmat=D_mat_assemble(p,n,n_al)
Dmat_inv=np.linalg.inv(Dmat)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Solving"
t_start=time.time()
tau0=np.dot(Dmat0_inv,eps0)
tau_grads=np.dot(Dmat_inv,eps_grads)
t_end=time.time()
print "dt = ", t_end-t_start





fname=''
fig=pl.figure()
ax = fig.add_subplot(1, 1, 1)
#
Dlims=[3*n_al*((r-1)*(r-1)+3*(r-1))/2 for r in range(1,p+2)]
plt=pl.imshow(Dmat,cmap=pl.cm.jet,interpolation='none',norm=mpl.colors.SymLogNorm(linthresh=.1,linscale=10,vmin=np.min(Dmat),vmax=np.max(Dmat)))
#plt=pl.imshow(Dmat,interpolation='none',norm=mpl.colors.SymLogNorm(linthresh=.001,linscale=1.,vmin=np.min(Dmat),vmax=np.max(Dmat)),origin='upper')
pl.colorbar(plt,ticks=[np.min(Dmat),0,np.max(Dmat)])
for i in range(len(Dlims)):
	pl.plot([Dlims[i],Dlims[i]],[0,Dlims[-1]],'k',lw=2.)
	pl.plot([0,Dlims[-1]],[Dlims[i],Dlims[i]],'k',lw=2.)
pl.xlim(Dlims[0],Dlims[-1])
pl.ylim(Dlims[0],Dlims[-1])
pl.xticks(Dlims)
pl.yticks(Dlims)
ax.xaxis.set_ticks_position('top')
ax.invert_yaxis()
pl.savefig('Dmat_'+fname+'.png')


xvals=np.arange(0,10,.01)
yvals=np.arange(0,10,.01)
xx,yy=np.meshgrid(xvals, yvals)
tau_ij=np.zeros((len(xvals),len(yvals),5))
sig_ij=np.zeros((len(xvals),len(yvals),5))

for i in range(len(xvals)):
	for j in range(len(yvals)):
		if (xvals[i]<=5):
			if (yvals[j]<=5):
				imat=0

			else:
				imat=3
		else:
			if (yvals[j]<=5):
				imat=1
			else:
				imat=2
		tau_ij[i,j][:3]=tau(p,tau0,tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
		tau_ij[i,j][4]=error_div(p,tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
		sig_ij[i,j][0]=mat[imat].L1111*tau_ij[i,j][0]+mat[imat].L1122*tau_ij[i,j][1]+2.*mat[imat].L1112*tau_ij[i,j][2]
		sig_ij[i,j][1]=mat[imat].L2211*tau_ij[i,j][0]+mat[imat].L2222*tau_ij[i,j][1]+2.*mat[imat].L2212*tau_ij[i,j][2]
		sig_ij[i,j][2]=mat[imat].L1211*tau_ij[i,j][0]+mat[imat].L1222*tau_ij[i,j][1]+2.*mat[imat].L1212*tau_ij[i,j][2]




		tau_dev_11=.5*(tau_ij[i,j][1]-tau_ij[i,j][0])
		#tau_dev_22=-tau_dev_11
		#tau_dev_12=tau_12
		tau_ij[i,j][3]=np.sqrt((tau_dev_11)**2+(-tau_dev_11)**2+2.*(tau_ij[i,j][2])**2)
#
name_ij=['11','22','12','vm','err']
ncols=45
cbar_label=[r'$(\tau_{11}/T_0)(T_0+T_1)/(T_0+2T_1)$',\
			r'$(\tau_{22}/T_0)(T_0+T_1)/(T_0+2T_1)$',
			r'$\tau_{12}/T_0$',\
	 r'$\|\boldsymbol{\tau}_{\mathrm{dev}2D}\|/T_0$',r'$\epsilon$']
cbar_label2=[r'$(\sigma_{11}/T_0)(T_0+T_1)/(T_0+2T_1)$',\
			r'$(\sigma_{22}/T_0)(T_0+T_1)/(T_0+2T_1)$',
			r'$\sigma_{12}/T_0$',\
	 r'$\|\boldsymbol{\tau}_{\mathrm{dev}2D}\|/T_0$',r'$\epsilon$']
cte_ij=[1./T0*(T0+T1)/(T0+2.*T1),1./T0*(T0+T1)/(T0+2.*T1),1./T0,1./T0,1.]
for ij in range(5):
	fig=pl.figure()
	org='image'
	vmax=cte_ij[ij]*np.max(tau_ij[:,:,ij])
	vmin=cte_ij[ij]*np.min(tau_ij[:,:,ij])
	if (ij<3):
		#max_abs=max(vmax,abs(vmin))
		#vmax=max_abs
		#vmin=-max_abs
		cntr=pl.contourf(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax)#,cmap=pl.cm.seismic)
		pl.contour(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
	else:
		cntr=pl.contourf(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax)
		pl.contour(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
	pl.plot((0,10),(5,5),'k')
	pl.plot((5,5),(0,10),'k')
	cbar=pl.colorbar(cntr)
	cbar.set_label(cbar_label[ij])
	pl.axes().get_xaxis().set_visible(False)
	pl.axes().get_yaxis().set_visible(False)
	pl.axes().set_aspect('equal')
	pl.savefig('tau'+name_ij[ij]+'_'+str(p)+'_'+str(n)+'_'+'.png',bbox_inches='tight')
	#
	if (ij<3):
		fig=pl.figure()
		org='image'
		vmax=cte_ij[ij]*np.max(sig_ij[:,:,ij])
		vmin=cte_ij[ij]*np.min(sig_ij[:,:,ij])
		if (ij<3):
			#max_abs=max(vmax,abs(vmin))
			#vmax=max_abs
			#vmin=-max_abs
			cntr=pl.contourf(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax)#,cmap=pl.cm.seismic)
			pl.contour(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)

		else:
			cntr=pl.contourf(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax)
			pl.contour(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		pl.plot((0,10),(5,5),'k')
		pl.plot((5,5),(0,10),'k')
		cbar=pl.colorbar(cntr)
		cbar.set_label(cbar_label2[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')
		pl.savefig('sig'+name_ij[ij]+'_'+str(p)+'_'+str(n)+'_'+'.png',bbox_inches='tight')




# TO DO:
# - There is no error when trying to compute T_infl(n,...) for n>6, there should be!
# - Speed-up code
# - MUST enforce condition adue to Maxwell-Betti theorem
#	- ... and carry it over for higher orders?



# See Eqs. III.r (r=1,...,p+1) on Slides 30, 31 and 32
# -> Global matrices form D^1_1 to D^p_p
#
# See Slide 26
# -> D^p_p requires global matrices M_{p,p} and T_{p,p}
#
# -> T_{p,p} is an assembly of {}^nT^{a,b}_{p,p}
#
# See Slide 19
# -> M_{p,p} requires Minkowski tensors W^{p+p,0}_0
#
# See Slide 11
# -> {}^nT^{a,b}_{p,p} requires Minkowski tensors W^{n+p,0}_0
# -> {}^nT^{a,b}_{p,p} requires gradients of the reference Green operator up to order n
#
# To sum up. Assume n>=p. Then,
# Compute	- Minkowski tensors up to W^{max(2p,n+p),0}_0
#			- Gradients of the reference Green operator up to order n




# kappa2D=lamda2D+mu2D
# E2D=4mu2D*(lambda2D+mu2D)/(lambda2D+2mu2D)
# mu2D=T0
# kappa2D=2.*T1
# E2D=4.*T0*(2.*T1+T0)/(2.*T1+2.*T0)
# E2D=2.*T0*(T0+2.*T1)/(T0+T1)
# Plot tau11/E2D, tau22/E2D and tau12/mu2d
