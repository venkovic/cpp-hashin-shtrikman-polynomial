import numpy as np

class medium2D:
	def __init__(self, isym):
		# 2D Anisotropic,         isym=0 
		# 2D Orthotropic,         isym=1 
		# 2D R0-Orthotropic,      isym=2 
		# 2D Square Symmetric,    isym=3 
		# 2D Polar Isotropic,     isym=4 
		# 2D Isotropic,           isym=5
		sym_labels=['2D_Anisotropic','2D_Orthotropic',\
					'2D_R0-Orthotropic','2D_Square_Symmetric',\
					'2D_Polar_Isotropic','2D_Isotropic']
		self.sym=sym_labels[isym]
		self.isym=isym
	#
	def set_stiff_polar(self, p):
		if (self.isym==0):
			self.P0=p[0]
			self.P1=p[1]
			self.R0=p[2]
			self.R1=p[3] 
			self.T0=p[4]
			self.T1=p[5]
			self.L1111=self.T0+2.*self.T1+self.R0*np.cos(4.*self.P0)+4.*self.R1*np.cos(2.*self.P1)
			self.L1112=self.R0*np.sin(4.*self.P0)+2.*self.R1*np.sin(2.*self.P1)
			self.L1122=-self.T0+2.*self.T1-self.R0*np.cos(4.*self.P0)
			self.L1212=self.T0-self.R0*np.cos(4.*self.P0)
			self.L2212=-self.R0*np.sin(4.*self.P0)+2.*self.R1*np.sin(2.*self.P1)
			self.L2222=self.T0+2.*self.T1+self.R0*np.cos(4.*self.P0)-4.*self.R1*np.cos(2.*self.P1)			
		elif (self.isym==1): 
			self.Th=p[0]
			self.K =p[1]
			self.R0=p[2]
			self.R1=p[3] 
			self.T0=p[4]
			self.T1=p[5]		
			self.L1111=self.T0+2*self.T1+(-1)**self.K*self.R0*np.cos(4*self.Th)+4*self.R1*np.cos(2*self.Th)
			self.L1112=-(-1)**self.K*self.R0*np.sin(4*self.Th)-2*self.R1*np.sin(2*self.Th)
			self.L1122=-self.T0+2*self.T1-(-1)**self.K*self.R0*np.cos(4*self.Th)
			self.L1212=self.T0-(-1)**self.K*self.R0*np.cos(4*self.Th)
			self.L2212=(-1)**self.K*self.R0*np.sin(4*self.Th)-2*self.R1*np.sin(2*self.Th)
			self.L2222=self.T0+2*self.T1+(-1)**self.K*self.R0*np.cos(4*self.Th)-4*self.R1*np.cos(2*self.Th)
		elif (self.isym==2): 
			self.Th=p[0]
			self.R1=p[1] 
			self.T0=p[2]
			self.T1=p[3]	
			self.L1111=self.T0+2*self.T1+4*self.R1*np.cos(2*self.Th)
			self.L1112=-2*self.R1*np.sin(2*self.Th)
			self.L1122=-self.T0+2*self.T1
			self.L1212=self.T0
			self.L2212=-2*self.R1*np.sin(2*self.Th)
			self.L2222=self.T0+2*self.T1-4*self.R1*np.cos(2*self.Th)
		elif (self.isym==3): 
			self.Th=p[0]
			self.R0=p[1]
			self.T0=p[2]
			self.T1=p[3]	
			self.L1111=self.T1+2*T1+self.R0*np.cos(4*self.Th)
			self.L1112=-self.R0*np.sin(4*self.Th)
			self.L1122=-self.T1+2*T1-self.R0*np.cos(4*self.Th)
			self.L1212=self.T1-self.R0*np.cos(4*self.Th)
			self.L2212=self.R0*np.sin(4*self.Th)
			self.L2222=self.T1+2*T1+self.R0*np.cos(4*self.Th)
		elif (self.isym==4): 
			return 0
		elif (self.isym==5): 
			self.k=p[0]
			self.m=p[1]
			self.L1111=self.k+self.m
			self.L1112=0.
			self.L1122=self.k-self.m
			self.L1212=self.m
			self.L2212=0.
			self.L2222=self.k+self.m	
		#
		self.L2211=self.L1122
		self.L1121=self.L1112;self.L1211=self.L1112;self.L2111=self.L1112
		self.L2221=self.L2212;self.L1222=self.L2212;self.L2122=self.L2212
		self.L1221=self.L1212;self.L2112=self.L1212;self.L2121=self.L1212	
		#
		self.L_mandel=np.array([[self.L1111,self.L1122,np.sqrt(2)*self.L1112],\
							    [self.L2211,self.L2222,np.sqrt(2)*self.L2212],\
							    [np.sqrt(2)*self.L1211,np.sqrt(2)*self.L1222,2.*self.L1212]])	
		#
		self.M_mandel=np.linalg.inv(self.L_mandel)	
	#
	def check_pos(self):
		# Verify positiveness of the strain energy and symmetry
		if (self.isym==0):
			flag1=True; flag2=True
			if not (self.T0-self.R0>0): flag1=False
			if not (self.T1*(self.T0**2-self.R0**2)-2.*self.R1**2*(self.T0-self.R0*np.cos(4.*(self.P0-self.P1)))>0): flag1=False
			if not (self.R0>=0): flag1=False
			if not (self.R1>=0): flag1=False
			if ((self.R0==0)|(self.R1==0)|(np.sin(4.*(self.P0-self.P1))==0)): flag2=False				
		elif (self.isym==1):
			flag1=True; flag2=True
			if not (self.T0-self.R0>0): flag1=False
			if not (self.T1*(self.T0+(-1.)**K*self.R0)-2.*self.R1**2>0): flag1=False
			if not (self.R0>=0): flag1=False
			if not (self.R1>=0): flag1=False
			if ((self.R0==0)|(self.R1==0)): flag2=False				
		elif (self.isym==2): 
			flag1=True; flag2=True
			if not (self.T0>0): flag1=False
			if not (self.T1*self.T0-2.*self.R1**2>0): flag1=False
			if not (self.R1>=0): flag1=False
			if (self.R1==0): flag2=False	
		elif (self.isym==3): 
			flag1=True; flag2=True
			if not (self.T0-self.R0>0): flag1=False
			if not (self.T1*(self.T0-self.R0)>0): flag1=False
			if not (self.R0>=0): flag1=False
			if (self.R0==0): flag2=False	
		elif (self.isym==4): 
			return 0
		elif (self.isym==5):
			if not (self.k>0): flag1=False
			if not (self.m>0): flag1=False
			 
			return 0
		#
		if (flag1&flag2): 
			return 0
		elif (not flag1):
			return 1 # Strain energy not positive
		elif (not flag2):
			return 2 # Symmetry not properly identified


















#
# Verify strain energy is positive and the symmetry is properly identified
def check_pos(mat):
	if (mat.isym==0):
		flag1=True
		flag2=True
		if not (mat.T0-mat.R0>0):
			flag1=False
		if not (mat.T1*(mat.T0**2-mat.R0**2)-2.*mat.R1**2*(mat.T0-mat.R0*np.cos(4.*(mat.P0-mat.P1)))>0): 
			flag1=False
		if not (mat.R0>=0): 
			flag1=False
		if not (mat.R1>=0): 
			flag1=False
		if ((mat.R0==0)|(mat.R1==0)|(np.sin(4.*(mat.P0-mat.P1))==0)): 
			flag2=False				
	elif (mat.isym==1):
		flag1=True
		flag2=True
		if not (mat.T0-mat.R0>0): 
			flag1=False
		if not (mat.T1*(mat.T0+(-1.)**mat.K*mat.R0)-2.*mat.R1**2>0): 
			flag1=False
		if not (mat.R0>=0): 
			flag1=False
		if not (mat.R1>=0): 
			flag1=False
		if ((mat.R0==0)|(mat.R1==0)): 
			flag2=False				
	elif (mat.isym==2): 
		flag1=True
		flag2=True
		if not (mat.T0>0): 
			flag1=False
		if not (mat.T1*mat.T0-2.*mat.R1**2>0): 
			flag1=False
		if not (mat.R1>=0): 
			flag1=False
		if (mat.R1==0): 
			flag2=False	
	elif (mat.isym==3): 
		flag1=True
		flag2=True
		if not (mat.T0-mat.R0>0): 
			flag1=False
		if not (mat.T1*(mat.T0-mat.R0)>0): 
			flag1=False
		if not (mat.R0>=0): 
			flag1=False
		if (mat.R0==0): 
			flag2=False	
	elif (mat.isym==4): 
		return 0
	elif (mat.isym==5):
		if not (mat.k>0): 
			flag1=False
		if not (mat.m>0): 
			flag1=False
	#
	if (flag1&flag2): 
		return 0
	elif (not flag1):
		return 1
	elif (not flag2):
		return 2


