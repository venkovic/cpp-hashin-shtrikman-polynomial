import pybind11
from distutils.core import setup, Extension
from distutils import sysconfig

#cpp_args=['-std=c++11']
cpp_args=['-Ofast','-std=c++11']

ext_modules=[Extension('HSPolynomial',\
			 ['dnGami.cpp','HS_pre-proc.cpp',\
        'HS_T_mat.cpp','HS_M_mat.cpp','HS_D_mat.cpp',\
        'Minkowski.cpp','HS_post-proc.cpp',\
        'etc.cpp','CompleteBarnettLothe.cpp',\
			  'dkN1_anisotropic.cpp','dkN2_anisotropic.cpp',\
			  'dkN1_orthotropic.cpp','dkN2_orthotropic.cpp',\
			  'dkN1_r0_orthotropic.cpp','dkN2_r0_orthotropic.cpp',\
			  'dkN1_square_symmetric.cpp','dkN2_square_symmetric.cpp',\
			  'dkN1_isotropic.cpp','dkN2_isotropic.cpp',\
			  'dG.cpp','set_sym.cpp','wrapper.cpp'],\
             include_dirs=[pybind11.get_include()],\
             language='c++',\
             extra_compile_args=cpp_args,),]

setup(
    name='HSPolynomial',
    version='1.0',
    author='Nicolas Venkovic',
    author_email='nvenkov1@jhu.edu',
    description='HSPolynomial',
    ext_modules=ext_modules,
)
