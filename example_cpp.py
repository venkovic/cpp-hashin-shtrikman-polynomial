import numpy as np
import scipy.linalg as lng
import pylab as pl
import matplotlib as mpl
from matplotlib import colors, ticker, cm
from scipy.integrate import quad
import scipy.special as spe
import HSPolynomial as HS
import GreenAnisotropic2D
import HSPolynomialpy as HSPolynomial
import time
import os

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



# Output file nomenclature:
# - HSPolynomial files:
#		Stress:			HSPoly_sig{ij}_{p}_{n}_{L0_code}_{eps11}_{eps22}_{eps12}_{self,sum0}_isym{isym}{fig_format}
#		Polarization:	HSPoly_tau{ij}_{p}_{n}_{L0_code}_{eps11}_{eps22}_{eps12}_{self,sum0}_isym{isym}{fig_format}
#		Strain:			HSPoly_sig{ij}_{p}_{n}_{L0_code}_{eps11}_{eps22}_{eps12}_{self,sum0}_isym{isym}{fig_format}
# - FFT files:
#		Stress:			FFT_sig{ij}_{eps11}_{eps22}_{eps12}_isym{isym}{fig_format}
#		Polarization:	FFT_tau{ij}_{L0_code}_{eps11}_{eps22}_{eps12}_isym{isym}{fig_format}
#		Strain:			FFT_sig{ij}_{eps11}_{eps22}_{eps12}_isym{isym}{fig_format}




isym=0
#
# Order of polynomial polarization field
p=5
#
# Order of the Taylor expansion
n=2

fname="cpp_0"+"isym"+str(isym); self_flag=0
#fname="cpp_self"+"isym"+str(isym); self_flag=1

if self_flag==0:
	self_code='sum0'
elif self_flag==1:
	self_code='self'

# Derine heterogeneous medium made of phases
n_al=4
mat=[]
#
params=[[.3,3.6,1.,.7,1.1,2.0],[.7,1.6,.4,.9,1.2,2.4],[.4,1.7,1.,.5,1.4,4.2],[.3,1.6,1.,.8,1.2,2.4]]; isym=0
#params=[[1.2,.3],[.7,.8],[2.,.9],[1.7,.6]]; isym=5
#
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
#
cte_PSD_Reuss=.36126 # Highest factor by which we multiply the Reuss stiffness estimate for SPSD
# Eventually, need to find a way to estimate the bounding stiffness
M0_Reuss_mandel=np.zeros((3,3))
for k in range(n_al):
	M0_Reuss_mandel+=mat[k].M_mandel
M0_Reuss_mandel/=n_al
L0_Reuss_mandel=np.linalg.inv(M0_Reuss_mandel)
L0_Reuss_mandel*=cte_PSD_Reuss
#
L0_code='wReuss'
#L0_code='wVoigt'

L0=L0_Reuss_mandel
if L0_code=='wReuss':
	L0=L0_Reuss_mandel
elif L0_code=='wVoigt':
	L0=L0_Voigt_mandel

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

#
# Compute compliance differences
dM_mandel=[]
for k in range(n_al):
	dM_mandel.append(np.linalg.inv(mat[k].L_mandel-mat0.L_mandel))



def recover_eps0(tau0):
	r_eps0=np.zeros(n_al*3)
	for al in range(n_al):
		r_eps0[al*3:(al+1)*3]=np.dot(dM_mandel[al],tau0[al*3:(al+1)*3])
	return r_eps0

def recover_eps_grads(tau_grads):
	tau_start=[0]+[3*n_al*((k-1)**2+3*(k-1))/2 for k in range(1,p+1)]	
	r_eps_grads=np.zeros(3*n_al*((p+1)**2+3*(p+1))/2)
	for r in range(1,p+1):
		for al in range(n_al):
			for k in range(r+1):
				r_eps_grads[tau_start[r]+3*al*(r+1)+3*r:tau_start[r]+3*al*(r+1)+3*(r+1)]=np.dot(dM_mandel[al],tau_grads[tau_start[r]+3*al*(r+1)+3*r:tau_start[r]+3*al*(r+1)+3*(r+1)])
	return r_eps_grads


def recover_sig0(eps0):
	r_sig0=np.zeros(n_al*3)
	for al in range(n_al):
		r_sig0[al*3:(al+1)*3]=np.dot(mat[al].L_mandel,eps0[al*3:(al+1)*3])
	return r_sig0

def recover_sig_grads(eps_grads):
	eps_start=[0]+[3*n_al*((k-1)**2+3*(k-1))/2 for k in range(1,p+1)]	
	r_sig_grads=np.zeros(3*n_al*((p+1)**2+3*(p+1))/2)
	for r in range(1,p+1):
		for al in range(n_al):
			for k in range(r+1):
				r_sig_grads[eps_start[r]+3*al*(r+1)+3*r:eps_start[r]+3*al*(r+1)+3*(r+1)]=np.dot(mat[al].L_mandel,eps_grads[eps_start[r]+3*al*(r+1)+3*r:eps_start[r]+3*al*(r+1)+3*(r+1)])
	return r_sig_grads




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
		for i in range(k+1):
			if not (i%2): 			# i   = 0,   2,     4,     6,     ..., n_i.
				ni1=k-i/2 		# ni1 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni2=k-ni1			# ni2 = 0   ,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			elif (i%2):				# i   = 1,   3,     5,     7,     ..., n_i
				ni2=k-(i-1)/2 	# ni2 = n_i, n_i-1, n_i-2, n_i-3, ..., (n_i+1)/2
				ni1=k-ni2 		# ni1 = 0,   1,     2,     3,     ..., n_i-(n_i+1)/2
			#
			# Get number of indices combinations used for Mandel representation
			fac_den=np.sqrt(spe.binom(k,ni1))
			#
			i_start=sum(tau_sizes[:k])
			if (ni1>0):
				fac1=spe.binom(k-1,ni1-1)/fac_den
				div_tau[0]+=k*fac1*tau_grads[i_start+al*(k+1)*3+i*3+0]*dx1**(ni1-1)*dx2**ni2
				div_tau[1]+=k*fac1*tau_grads[i_start+al*(k+1)*3+i*3+2]/np.sqrt(2)*dx1**(ni1-1)*dx2**ni2
			if (ni2>0):
				fac2=spe.binom(k-1,ni2-1)/fac_den
				div_tau[0]+=k*fac1*tau_grads[i_start+al*(k+1)*3+i*3+2]/np.sqrt(2)*dx1**ni1*dx2**(ni2-1)
				div_tau[1]+=k*fac1*tau_grads[i_start+al*(k+1)*3+i*3+1]*dx1**ni1*dx2**(ni2-1)
	return np.sqrt(div_tau[0]**2+div_tau[1]**2)













#	
# Set up volume fractions
cfrac=np.array(n_al*[1./n_al])
#
# Set up centers of gravity
L=10.
xloc=[np.array((2.5,2.5)),np.array((7.5,2.5)),np.array((7.5,7.5)),np.array((2.5,7.5))]
#
mat_HS=HS.heterogeneous_medium()
mat_HS.set_mat(L, n_al, xloc, cfrac, dM_mandel)
mat_HS.set_p(p)
mat_HS.set_n(n)
print "\n Pre-processing for reference medium"
t_start=time.time()
mat_HS.set_ref(isym, p0)
t_end=time.time()
print "dt = ", t_end-t_start

eps11=0.; eps22=1.; eps12=0.
eps_av_mandel=np.array([eps11,eps22,np.sqrt(2)*eps12])
load=str(eps11)+'_'+str(eps22)+'_'+str(eps12)


print "\n Computing eps0"
t_start=time.time()
eps0=mat_HS.eps_global0(eps_av_mandel)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing eps_grads"
t_start=time.time()
eps_grads=mat_HS.eps_global(eps_av_mandel)
t_end=time.time()
print "dt = ", t_end-t_start

print "\n Computing Dmat0"
t_start=time.time()
Dmat0=mat_HS.D_mat0_assemble(self_flag)
Dmat0_inv=np.linalg.inv(Dmat0)
t_end=time.time()
#print Dmat0
print "dt = ", t_end-t_start

print "\n Computing Dmat"
t_start=time.time()
Dmat =mat_HS.D_mat_assemble(self_flag)
Dmat_inv=np.linalg.inv(Dmat)
t_end=time.time()
print "dt = ", t_end-t_start



# From np.array to cpp vector
# From cpp vector to np.array

# -Set-up L, n_al, xloc, cfrac, L_al
# -Set-up mat0_Green
#	-Compute dM
# -Compute eps0, eps_grads
# -Compute Dmat0, Dmat


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
pl.savefig('Dmat_'+str(p)+'_'+str(n)+'_'+L0_code+'_'+load+'_'+self_code+'_isym'+str(isym)+fig_format)

#########################################
#			PYTHON POST-PROC			#
#########################################
print "\n Solving"
t_start=time.time()
tau0=np.dot(Dmat0_inv,eps0)
tau_grads=np.dot(Dmat_inv,eps_grads)
t_end=time.time()
print "dt = ", t_end-t_start



# Recover polynomial strain field
#r_eps0=recover_eps0(tau0)
#r_eps_grads=recover_eps_grads(tau_grads)
# Recover polynomial stress field
#r_sig0=recover_sig0(eps0)
#r_sig_grads=recover_sig_grads(eps_grads)


print "\n Post-processing"
t_start=time.time()
#

xvals=np.arange(0,10,.05)
yvals=np.arange(0,10,.05)
xx,yy=np.meshgrid(xvals, yvals)
tau_ij=np.zeros((len(xvals),len(yvals),5))
sig_ij=np.zeros((len(xvals),len(yvals),5))
eps_ij=np.zeros((len(xvals),len(yvals),3))


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
		#tau_ij[i,j][:3]=tau(p,tau0,tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
		#tau_ij[i,j][4]=error_div(p,tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
	#	print "test a:", i,j
	#	tau_temp=np.random.rand(3)
		tau_temp=mat_HS.tau(tau0,tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
	#	print "test b."
		tau_ij[i][j][0]=tau_temp[0]
		tau_ij[i][j][1]=tau_temp[1]
		tau_ij[i][j][2]=tau_temp[2]
		#eps_ij[i,j][:3]=mat_HS.tau(r_eps0,r_eps_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
	#	print "test a:", i,j		
	#	print tau_grads.shape, imat, xvals[i], xloc[imat][0], yvals[j], xloc[imat][1]
	#	print tau_ij[i][j][4], tau_grads[10]	
		tau_ij[i][j][4]=mat_HS.error_div(tau_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
	#	test=.2
	#	print test
	#	tau_ij[i][j][4]=np.random.rand()
	#	print "test b."
		#sig_ij[i,j][:3]=mat_HS.tau(r_sig0,r_sig_grads,imat,xvals[i]-xloc[imat][0],yvals[j]-xloc[imat][1])
		eps_ij[i][j][0]=dM_mandel[imat][0][0]*tau_ij[i][j][0]+dM_mandel[imat][0][1]*tau_ij[i][j][1]+dM_mandel[imat][0][2]*np.sqrt(2.)*tau_ij[i][j][2]
		eps_ij[i][j][1]=dM_mandel[imat][1][0]*tau_ij[i][j][0]+dM_mandel[imat][1][1]*tau_ij[i][j][1]+dM_mandel[imat][1][0]*np.sqrt(2.)*tau_ij[i][j][2]
		eps_ij[i][j][2]=dM_mandel[imat][2][0]/np.sqrt(2.)*tau_ij[i][j][0]+dM_mandel[imat][2][1]/np.sqrt(2.)*tau_ij[i][j][1]+np.sqrt(2.)*dM_mandel[imat][2][2]*tau_ij[i][j][2]
		#
		sig_ij[i][j][0]=mat[imat].L1111*eps_ij[i][j][0]+mat[imat].L1122*eps_ij[i][j][1]+2.*mat[imat].L1112*eps_ij[i][j][2]
		sig_ij[i][j][1]=mat[imat].L2211*eps_ij[i][j][0]+mat[imat].L2222*eps_ij[i][j][1]+2.*mat[imat].L2212*eps_ij[i][j][2]
		sig_ij[i][j][2]=mat[imat].L1211*eps_ij[i][j][0]+mat[imat].L1222*eps_ij[i][j][1]+2.*mat[imat].L1212*eps_ij[i][j][2]
		tau_dev_11=.5*(tau_ij[i][j][1]-tau_ij[i][j][0])
		#tau_dev_22=-tau_dev_11
		#tau_dev_12=tau_12
		tau_ij[i][j][3]=np.sqrt((tau_dev_11)**2+(-tau_dev_11)**2+2.*(tau_ij[i][j][2])**2)
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
cbar_label3=[r'$\varepsilon_{11}$',\
			 r'$\varepsilon_{22}$',\
			 r'$\varepsilon_{12}$']
cte_ij=[1./T0*(T0+T1)/(T0+2.*T1),1./T0*(T0+T1)/(T0+2.*T1),1./T0,1./T0,1.]
for ij in range(5):
	fig=pl.figure()
	org='lower'
	vmax=cte_ij[ij]*np.max(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))]))
	vmin=cte_ij[ij]*np.min(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))]))
	if (ij<3):
		#max_abs=max(vmax,abs(vmin))
		#vmax=max_abs
		#vmin=-max_abs
		cntr=pl.contourf(xx,yy,cte_ij[ij]*np.transpose(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))])),ncols,origin=org,vmin=vmin,vmax=vmax)#,cmap=pl.cm.seismic)
		#pl.contour(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		pl.contour(xx,yy,cte_ij[ij]*np.transpose(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))])),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
	else:
		cntr=pl.contourf(xx,yy,cte_ij[ij]*np.transpose(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))])),ncols,origin=org,vmin=vmin,vmax=vmax)
		#pl.contour(xx,yy,cte_ij[ij]*tau_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		pl.contour(xx,yy,cte_ij[ij]*np.transpose(np.array([[tau_ij[vi][vj][ij] for vj in range(len(tau_ij[0]))] for vi in range(len(tau_ij))])),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:2]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
	pl.plot((0,10),(5,5),'k')
	pl.plot((5,5),(0,10),'k')
	cbar=pl.colorbar(cntr)
	cbar.set_label(cbar_label[ij])
	pl.axes().get_xaxis().set_visible(False)
	pl.axes().get_yaxis().set_visible(False)
	pl.axes().set_aspect('equal')
	pl.savefig('tau'+name_ij[ij]+'_'+str(p)+'_'+str(n)+'_'+L0_code+'_'+load+'_'+self_code+'_isym'+str(isym)+fig_format,bbox_inches='tight')
	#
	if (ij<3):
		fig=pl.figure()
		vmax=cte_ij[ij]*np.max(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))]))
		vmin=cte_ij[ij]*np.min(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))]))
		if (ij<3):
			#max_abs=max(vmax,abs(vmin))
			#vmax=max_abs
			#vmin=-max_abs
			cntr=pl.contourf(xx,yy,cte_ij[ij]*np.transpose(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))])),ncols,origin=org,vmin=vmin,vmax=vmax)#,cmap=pl.cm.seismic)
			#pl.contour(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
			pl.contour(xx,yy,cte_ij[ij]*np.transpose(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))])),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)

		else:
			cntr=pl.contourf(xx,yy,cte_ij[ij]*np.transpose(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))])),ncols,origin=org,vmin=vmin,vmax=vmax)
			#pl.contour(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
			pl.contour(xx,yy,cte_ij[ij]*np.transpose(np.array([[sig_ij[vi][vj][ij] for vj in range(len(sig_ij[0]))] for vi in range(len(sig_ij))])),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:2]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)

		pl.plot((0,10),(5,5),'k')
		pl.plot((5,5),(0,10),'k')
		cbar=pl.colorbar(cntr)
		cbar.set_label(cbar_label2[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')
		pl.savefig('sig'+name_ij[ij]+'_'+str(p)+'_'+str(n)+'_'+L0_code+'_'+load+'_'+self_code+'_isym'+str(isym)+fig_format,bbox_inches='tight')


		fig=pl.figure()
		vmax=np.max(np.array([[eps_ij[vi][vj][ij] for vj in range(len(eps_ij[0]))] for vi in range(len(eps_ij))]))
		vmin=np.min(np.array([[eps_ij[vi][vj][ij] for vj in range(len(eps_ij[0]))] for vi in range(len(eps_ij))]))
		if (ij<3):
			#max_abs=max(vmax,abs(vmin))
			#vmax=max_abs
			#vmin=-max_abs
			cntr=pl.contourf(xx,yy,np.transpose(np.array([[eps_ij[vi][vj][ij] for vj in range(len(eps_ij[0]))] for vi in range(len(eps_ij))])),ncols,origin=org,vmin=vmin,vmax=vmax)#,cmap=pl.cm.seismic)
			#pl.contour(xx,yy,cte_ij[ij]*sig_ij[:,:,ij],ncols,origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
			pl.contour(xx,yy,np.transpose(np.array([[eps_ij[vi][vj][ij] for vj in range(len(eps_ij[0]))] for vi in range(len(eps_ij))])),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)

		pl.plot((0,10),(5,5),'k')
		pl.plot((5,5),(0,10),'k')
		cbar=pl.colorbar(cntr)
		cbar.set_label(cbar_label3[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')
		pl.savefig('eps'+name_ij[ij]+'_'+str(p)+'_'+str(n)+'_'+L0_code+'_'+load+'_'+self_code+'_isym'+str(isym)+fig_format,bbox_inches='tight')



t_end=time.time()
print "dt = ", t_end-t_start


pl.close('all')

#
# Check accuracy of the Taylor expansion used in place of the Green operator for strains
if True:
	print "\n Checking accuracy of Taylor expansion:"

	def G(ijkl,r,th):
		return HS.dnGami(mat_HS.mat0_Green,0,ijkl,r,th)
	def dnG(n,ijkl,n1,r,th):
		return HS.dnGami(mat_HS.mat0_Green,n,ijkl+n1*[1]+(n-n1)*[2],r,th)


	def n_Taylor(n,ijkl,x_al,x_ga,x,y):
		# x is relative to x_al
		# y is relative to x_ga
		x_ga_al=x_al-x_ga
		r=np.sqrt(x_ga_al[0]**2+x_ga_al[1]**2)
		th=np.arctan2(x_ga_al[1],x_ga_al[0])
		fn=dnG(0,ijkl,0,r,th)
		for k in range(1,n+1):
			for i in range(0,k+1):
				k_i_A_ijkl_n1_al_n1_ga=0.
				for n1_al in range(0,k-i+1):
					fac_al=spe.binom(k-i+1,n1_al)
					for n1_ga in range(0,i+1):
						fac_ga=spe.binom(i+1,n1_ga)
						k_i_A_ijkl_n1_al_n1_ga+=fac_al*fac_ga*dnG(k,ijkl,n1_al+n1_ga,r,th)*x[0]**n1_al*x[1]**(k-i-n1_al)*y[0]**n1_ga*y[1]**(i-n1_ga)
				fn+=(-1)**i/spe.factorial(k-i)/spe.factorial(i)*k_i_A_ijkl_n1_al_n1_ga
		return fn

	ijkl=[1,2,1,2]

	x_al=np.array([2.5,2.5])
	x_ga=np.array([7.5,2.5])
	x_ga_al=x_al-x_ga

	# Relative coordinates:
	#x=np.array([2.5,-2.5])
	x=np.array([0,0])
	Y1=np.linspace(-2.5,2.5,101)
	y2=-2.5	
	#y2=0.

	GreenTaylor={}
	for k in range(1,5):
		GreenTaylor[k]=np.array([n_Taylor(k,ijkl,x_al,x_ga,x,np.array([y1,y2])) for y1 in Y1])


	

	Green_=np.array([G(ijkl,np.linalg.norm(x_ga_al+x-np.array([y1,y2])),np.arctan2(x_ga_al[1]+x[1]-y2,x_ga_al[0]+x[0]-y1)) for y1 in Y1])


	labs=[r'${}^{(0)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$',
		  r'${}^{(1)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$',
		  r'${}^{(2)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$',
		  r'${}^{(3)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$',
		  r'${}^{(4)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$',
		  r'${}^{(5)}\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$']
	fig=pl.figure()
	pl.plot(Y1,Green_,'k-',label=r'$\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$')
	for k in range(1,5):
		#pl.semilogy(Y1,GreenTaylor[k],label=labs[k])
		pl.plot(Y1,GreenTaylor[k],label=labs[k])
	pl.plot([0,0],Green_[:2],lw=0.,label=r'$\underline{x}=2.5(\underline{e}_1-\underline{e}_2)$')
	#pl.plot([0,0],Green_[:2],lw=0.,label=r'$\underline{x}=\underline{0}$')
	pl.plot([0,0],Green_[:2],lw=0.,label=r'$\underline{y}=y_1\underline{e}_1-2.5\underline{e}_2$')
	#pl.plot([0,0],Green_[:2],lw=0.,label=r'$\underline{y}=y_1\underline{e}_1$')
	pl.legend(loc=1)
	pl.xlabel(r'$y_1$')
	#pl.xlim(-2.5,2.5)
	pl.axes().set_aspect('equal')		
	pl.xlim(-2.5,2.5)
	pl.ylim(0,1.)
	pl.ylabel(r'$\Gamma_{1212}(\underline{x}-\underline{y}+\underline{x}_{\gamma\alpha})$')
#	pl.savefig('Inaccurate_Taylor_for_Green2_'+str(isym)+fig_format,bbox_inches='tight')
	pl.show()









	import scipy.integrate as intg
	intg.quad(lambda th: G([1,2,1,2],1.,th)*np.sin(th)**2, 0, 2.*np.pi)

	intg.quad(lambda th: (mat0_Green.dnGi(1,[1,2,2],.4,th)+mat0_Green.dnGi(1,[2,2,1],.4,th))*np.cos(th), 0, 2.*np.pi)
	intg.quad(lambda th: (mat0_Green.dnGi(2,[1,2,1,2],.4,th)*(np.sin(th)+th)), 0, 2.*np.pi)


	#
	# Fourier series representation of angular component of Green operator
	an=[0.]
	bn=[0.]
	n_=80
	# Period is T=np.pi
	T=2.*np.pi
	for k in range(1,n_+1):
		an+=[intg.quad(lambda th: G([1,2,1,2],1.,th)/(T/2.)*np.cos(k*2*np.pi*th/T), -T/2., T/2.)[0]]
		bn+=[intg.quad(lambda th: G([1,2,1,2],1.,th)/(T/2.)*np.sin(k*2*np.pi*th/T), -T/2., T/2.)[0]]

	def Fourier_G(r,th,n):
		return (sum([an[k]*np.cos(k*2*np.pi*th/T) for k in range(1,n+1)])+sum([bn[k]*np.sin(k*2*np.pi*th/T) for k in range(1,n+1)]))/r**2



	ths=np.linspace(0,	2.*np.pi,1000)
	pl.figure()
	pl.plot(ths,[G([1,2,1,2],1.,th) for th in ths])
	pl.plot(ths,[Fourier_G(1.,th,n_) for th in ths])
	pl.xlim(0,2.*np.pi)
	pl.show()



#
# Get a Mandel representation of the stiffness
def get_Lmat(mat):
	Lmat=np.array([[mat.L1111,mat.L1122,np.sqrt(2)*mat.L1112],
				  [mat.L2211,mat.L2222,np.sqrt(2)*mat.L2212],
				  [np.sqrt(2)*mat.L1211,np.sqrt(2)*mat.L1222,2.*mat.L1212]])
	return Lmat
#	
# Generalized Young's modulus
def gen_E(th,mat,Smat):
	m2vec=np.array([np.cos(th)**2,np.sin(th)**2,np.sqrt(2)*np.cos(th)*np.sin(th)])
	return np.dot(np.dot(m2vec,Smat),m2vec)**-1
#
# Generalized shear modulus
def gen_mu(th,mat,Smat):
	mbyp_vec=np.array([np.cos(th)*(-np.sin(th)),np.sin(th)*np.cos(th),np.sqrt(2)*np.cos(th)*np.cos(th)])
	mbyp_vec+=np.array([np.cos(th)*(-np.sin(th)),np.sin(th)*np.cos(th),np.sqrt(2)*np.sin(th)*(-np.sin(th))])
	mbyp_vec/=2.
	return np.dot(4.*np.dot(mbyp_vec,Smat),mbyp_vec)**-1
#
# Generalized Poisson's ratio
def gen_nu(th,mat,Smat):
	m2vec=np.array([np.cos(th)**2,np.sin(th)**2,np.sqrt(2)*np.cos(th)*np.sin(th)])
	p2vec=np.array([(-np.sin(th))**2,np.cos(th)**2,np.sqrt(2)*(-np.sin(th))*np.cos(th)])
	return -np.dot(np.dot(p2vec,Smat),m2vec)/np.dot(np.dot(m2vec,Smat),m2vec)
#
# Generalized absoulte value of Poisson's ratio
def gen_nu(th,mat,Smat):
	m2vec=np.array([np.cos(th)**2,np.sin(th)**2,np.sqrt(2)*np.cos(th)*np.sin(th)])
	p2vec=np.array([(-np.sin(th))**2,np.cos(th)**2,np.sqrt(2)*(-np.sin(th))*np.cos(th)])
	return -np.dot(np.dot(p2vec,Smat),m2vec)/np.dot(np.dot(m2vec,Smat),m2vec)
#

fig_format='.eps'
def plot_polar(mat,fname,add_line_flag,angle):
	nvals=1000
	tvals=np.linspace(0,2.*np.pi,nvals)		
	#
	Lmat=get_Lmat(mat)
	Smat=lng.inv(Lmat)
	E_iso=quad(gen_E,0,2.*np.pi,args=(mat,Smat))[0]/2./np.pi
	E_vals=np.array([gen_E(t,mat,Smat) for t in tvals])/E_iso
	mu_iso=quad(gen_mu,0,2.*np.pi,args=(mat,Smat))[0]/2./np.pi
	mu_vals=np.array([gen_mu(t,mat,Smat) for t in tvals])/mu_iso
	nu_iso=quad(gen_nu,0,2.*np.pi,args=(mat,Smat))[0]/2./np.pi
	nu_vals=np.array([gen_nu(t,mat,Smat) for t in tvals])/nu_iso
	#
	fig=pl.figure()
	ax = pl.subplot(111, projection='polar')
	ax.plot(tvals, E_vals,lw=1.5,label=r'$E(\theta)/E$')
	ax.plot(tvals, mu_vals,lw=1.5,label=r'$\mu(\theta)/\mu$')
	ax.plot(tvals[nu_vals>=0], nu_vals[nu_vals>=0],lw=1.5,label=r'$\nu(\theta)/\nu$')
	ax.plot(tvals[nu_vals<0], nu_vals[nu_vals<0],lw=1.5)
	max_mag=max(1.1*max(E_vals),1.1*max(mu_vals),1.1*max(nu_vals))
	if add_line_flag:
		ax.plot(2*[angle],[0,max_mag],lw=2.,color='k')
	ax.set_rmax(max_mag)
	pl.legend()
	ax.set_xticklabels([])
	pl.savefig('figPolar'+fname+fig_format,bbox_inches='tight')
	pl.close(fig)

if False:
	plot_polar(mat[0],'isym0_al0',False,.5)
	plot_polar(mat[1],'isym0_al1',False,.5)
	plot_polar(mat[2],'isym0_al2',False,.5)
	plot_polar(mat[3],'isym0_al3',False,.5)




ths=np.linspace(0,2.*np.pi,1000)
# Return components of translated Minkowski tensors
def W(r,ni1,ni2):
	# ni1+ni2=r
	a=1/2.
	Wr_i=((a/2.)**(ni1+ni2+2)\
		 -(-a/2.)**(ni1+1)*(a/2.)**(ni2+1)\
		 -(a/2.)**(ni1+1)*(-a/2.)**(ni2+1)\
		 +(-a/2.)**(ni1+1)*(-a/2.)**(ni2+1))/(ni1+1)/(ni2+1)
	return Wr_i
#	
def Reynolds_W(s,ths):	
	xvals=np.zeros(len(ths))
	yvals=np.zeros(len(ths))
	for j in range(s+1):
		if not (j%2): 		# j   = 0, 2,   4,   6,   ..., s
			nj1=s-j/2   	# nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
			nj2=s-nj1		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
		#
		elif (j%2): 		# j   = 0, 2,   4,   6,   ..., s
			nj2=s-(j-1)/2   # nj1 = s, s-1, s-2, s-3, ..., (s+1)/2
			nj1=s-nj2		# nj2 = 0, 1,   2,   3,   ..., s-(s+1)/2
		val=spe.binom(s,nj1)*W(s,nj1,nj2)
		#val=W(s,nj1,nj2)
		xvals+=val*np.cos(ths)**(nj1+1)*np.sin(ths)**nj2
		yvals+=val*np.cos(ths)**(nj1)*np.sin(ths)**(nj2+1)
		
	return np.sqrt(xvals**2+yvals**2)


if False:
	nn=12
	fig=pl.figure()
	ax = pl.subplot(111, projection='polar')
	for s in range(1,nn+1):
		rvals=Reynolds_W(s,ths)
		rvals/=np.max(rvals)
		ax.plot(ths,rvals,lw=1.5)
	ax.set_xticklabels([])
	pl.savefig('figPolar_W2'+fig_format,bbox_inches='tight')
	pl.close(fig)



#
# Read output file of scalar stored in row-major
def read_T0(file_name):
	file_in=open(file_name,'read')
	spamreader=file_in.readlines()
	file_in.close()
	spamrow=spamreader[0].split(',')
	nx=int(spamrow[0])
	ny=int(spamrow[1])
	T0=np.zeros((nx,ny))
	for k in range(nx*ny):
		i=k/nx
		j=k%nx
		T0[i,j]=float(spamreader[k+1])
	return T0
#
# Read output file of 2nd order tensors stored in row-major
def read_T2(file_name):
	file_in=open(file_name,'read')
	spamreader=file_in.readlines()
	file_in.close()
	spamrow=spamreader[0].split(',')
	nx=int(spamrow[0])
	ny=int(spamrow[1])
	T2_11=np.zeros((nx,ny))
	T2_22=np.zeros((nx,ny))
	T2_12=np.zeros((nx,ny))
	for k in range(nx*ny):
		spamrow=spamreader[k+1].split()
		spamrow=spamrow[0].split(',')
		i=k/nx
		j=k%nx
		T2_11[i,j]=float(spamrow[0])
		T2_22[i,j]=float(spamrow[1])
		T2_12[i,j]=float(spamrow[2])
	return T2_11, T2_22, T2_12

#
# Read output file of error in equilibrium
def read_err(fname):
	file_in=open(fname,'read')
	spamreader=file_in.readlines()
	file_in.close()
	niter=int(spamreader[0])
	err=np.zeros(niter+1)
	for k in range(niter+1):
		err[k]=float(spamreader[k+1])
	return err


#
# Write geometry file // USNCCM
def write_geo_USNCCM(nx,ny,fname):
	file_out=open(fname,'write')
	file_out.write(str(nx)+','+str(ny)+'\n')
	for i in range(1,nx+1):
		for j in range(1,ny+1):
			x=(i-.5)/nx
			y=(j-.5)/ny
			if (x<.5):
				if (y<.5):
					file_out.write('0\n')
				else:
					file_out.write('3\n')
			else:
				if (y<.5):
					file_out.write('1\n')
				else:
					file_out.write('2\n')
	file_out.close()
#
# Write material file // Willot (2015)
def write_mat_USNCCM(fname):
	#
	L1111=[mat[i].L1111 for i in range(n_al)]
	L1122=[mat[i].L1122 for i in range(n_al)]
	L1112=[mat[i].L1112 for i in range(n_al)]
	L2222=[mat[i].L2222 for i in range(n_al)]
	L2212=[mat[i].L2212 for i in range(n_al)]
	L1212=[mat[i].L1212 for i in range(n_al)]
	#
	file_out=open(fname,'write')
	file_out.write(str(n_al)+'\n')
	for i in range(n_al):
		file_out.write(str(L1111[i])+','+str(L1122[i])+','+str(L1112[i])+','+str(L2222[i])+','+str(L2212[i])+','+str(L1212[i])+'\n')
	file_out.close()
	return L1111, L1122, L1112, L2222, L2212, L1212


if False:
	write_geo_USNCCM(1024,1024,'USNCCM_1024.geo')
	L1111, L1122, L1112, L2222, L2212, L1212 = write_mat_USNCCM('USNCCM.mat')
	eps11=0.; eps22=1.; eps12=0.
	#
	# Set up tolerance for error in equlibrium
	etol=10.**-11
	#
	# Run spectral solver
	#out=os.system('./fft_solver 1024 '+str(mat0.L1111)+' '+str(mat0.L1122)+' '+str(mat0.L1112)+' '+str(mat0.L2222)+' '+str(mat0.L2212)+' '+str(mat0.L1212)+' '+str(eps11)+' '+str(eps22)+' '+str(eps12)+' '+str(etol)+' USNCCM_1024.geo USNCCM.mat USNCCM_1024')
	out=os.system('./fft_solver 1024 '+str(np.sum(L1111))+' '+str(np.sum(L1122))+' '+str(np.sum(L1112))+' '+str(np.sum(L2222))+' '+str(np.sum(L2212))+' '+str(np.sum(L1212))+' '+str(eps11)+' '+str(eps22)+' '+str(eps12)+' '+str(etol)+' USNCCM_1024.geo USNCCM.mat USNCCM_1024')
	#
	# Read solver outputs
	m_sig_11_1024, m_sig_22_1024, m_sig_12_1024 = read_T2('USNCCM_1024.sig_ij')
	m_eps_11_1024, m_eps_22_1024, m_eps_12_1024 = read_T2('USNCCM_1024.eps_ij')
	e_1024=read_err('USNCCM_1024.err')
	#
	Nx=1024; Ny=1024
	Lx=1.; Ly=1.
	x1=Lx/Nx*np.array([i-1/2. for i in range(1,Nx+1)])
	x2=Ly/Ny*np.array([i-1/2. for i in range(1,Ny+1)])



	tau_11_1024=m_sig_11_1024-mat0.L1111*m_eps_11_1024-mat0.L1122*m_eps_22_1024-2.*mat0.L1112*m_eps_12_1024
	tau_22_1024=m_sig_22_1024-mat0.L2211*m_eps_11_1024-mat0.L2222*m_eps_22_1024-2.*mat0.L2212*m_eps_12_1024
	tau_12_1024=m_sig_12_1024-mat0.L1211*m_eps_11_1024-mat0.L1222*m_eps_22_1024-2.*mat0.L1212*m_eps_12_1024

	data_sig_ij=np.array([m_sig_11_1024,m_sig_22_1024,m_sig_12_1024])
	data_eps_ij=np.array([m_eps_11_1024,m_eps_22_1024,m_eps_12_1024])
	data_tau_ij=np.array([tau_11_1024,tau_22_1024,tau_12_1024])
	

	fig_format='.png'

	ncols=45
	org='lower'
	for ij in range(3):	
		fig=pl.figure()
		vmax=cte_ij[ij]*np.max(data_sig_ij[ij])
		vmin=cte_ij[ij]*np.min(data_sig_ij[ij])
		cntr=pl.contourf(x1,x2,cte_ij[ij]*np.transpose(data_sig_ij[ij]),ncols,origin=org,vmin=vmin,vmax=vmax)
		pl.contour(x1,x2,cte_ij[ij]*np.transpose(data_sig_ij[ij]),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		cbar=pl.colorbar(cntr)
		pl.plot((0,1),(.5,.5),'-k',lw=2)
		pl.plot((.5,.5),(0,1),'-k',lw=2)
		cbar.set_label(cbar_label2[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')		
		pl.savefig('FFT_sig'+name_ij[ij]+'_'+load+'_isym'+str(isym)+fig_format,bbox_inches='tight')
		#
		fig=pl.figure()
		vmax=cte_ij[ij]*np.max(data_tau_ij[ij])
		vmin=cte_ij[ij]*np.min(data_tau_ij[ij])
		cntr=pl.contourf(x1,x2,cte_ij[ij]*np.transpose(data_tau_ij[ij]),ncols,origin=org,vmin=vmin,vmax=vmax)
		pl.contour(x1,x2,cte_ij[ij]*np.transpose(data_tau_ij[ij]),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		cbar=pl.colorbar(cntr)
		pl.plot((0,1),(.5,.5),'-k',lw=2)
		pl.plot((.5,.5),(0,1),'-k',lw=2)
		cbar.set_label(cbar_label[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')
		pl.savefig('FFT_tau'+name_ij[ij]+'_'+L0_code+'_'+load+'_isym'+str(isym)+fig_format,bbox_inches='tight')
		#
		fig=pl.figure()
		vmax=np.max(data_eps_ij[ij])
		vmin=np.min(data_eps_ij[ij])
		cntr=pl.contourf(x1,x2,np.transpose(data_eps_ij[ij]),ncols,origin=org,vmin=vmin,vmax=vmax)
		pl.contour(x1,x2,np.transpose(data_eps_ij[ij]),levels=np.concatenate([cntr.levels[:-30],cntr.levels[-30:-1:1]]),origin=org,vmin=vmin,vmax=vmax, colors='black', linewidth=.2)
		cbar=pl.colorbar(cntr)
		pl.plot((0,1),(.5,.5),'-k',lw=2)
		pl.plot((.5,.5),(0,1),'-k',lw=2)
		cbar.set_label(cbar_label2[ij])
		pl.axes().get_xaxis().set_visible(False)
		pl.axes().get_yaxis().set_visible(False)
		pl.axes().set_aspect('equal')
		pl.savefig('FFT_eps'+name_ij[ij]+'_'+load+'_isym'+str(isym)+fig_format,bbox_inches='tight')

