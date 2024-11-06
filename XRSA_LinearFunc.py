import numpy as np
from numpy import inf
from func_lib import *
from scipy.interpolate import interp1d
from numba import cuda
import scipy.integrate as integrate
import copy
from scipy.special import voigt_profile
from scipy.optimize import fsolve
from scipy.signal import fftconvolve
from scipy import interpolate
import traceback

class XRSA_LinearFunc():
    def __init__(self,func,lower_limit=-np.inf,upper_limit=np.inf) -> None:
        self.func = func
        self.domain = np.array([lower_limit,upper_limit]) #To simplify the integration
        
    def __call__(self, x):
        if x>self.domain[0] and x<self.domain[1]:
            return self.func(x)
        return 0
    
    def __add__(self,add_func:'XRSA_LinearFunc'):
        # print('called:',self)
        new_func = copy.deepcopy(self)
        if not isinstance(add_func,XRSA_Zero_Linear_Func):
            # print('calling_adding',self)
            original_func = new_func.func
            new_func.func = lambda x: original_func(x)+add_func.func(x)
            # print('called_adding',self)
            # print(new_func.domains,add_func.domains)
            new_func.domain = np.array([np.minimum(self.domain[0],add_func.domain[0]),np.maximum(self.domain[1],add_func.domain[1])])
            # new_func.Merge_Domains()
        return new_func
        
    def __mul__(self,scalar):
        # print('called',self)
        if scalar==0:
            return XRSA_Zero_Linear_Func()
        else:
            new_func = copy.deepcopy(self)
            original_func = new_func.func
            new_func.func = lambda x:scalar*original_func(x)
            return new_func
    
    def Normalization(self):
        integ = self.Total_Integration()
        func = self.func
        self.func = lambda x: func(x)/integ
    
    def Total_Integration(self):
        # self.Merge_Values()
        total_integ,_ = integrate.quad(lambda x: self(x),self.domain[0],self.domain[1],epsabs=1e-5,epsrel=1e-5)
        return total_integ
    
    def Get_moments(self,order=1,debug=False):
        # epsilon = (self.domain[1]-self.domain[0])*1e-6
        if debug:
            plot_func = XRSA_LinearFunc(lambda x: x * self(x), self.domain[0], self.domain[1])
            plot_func.plot_func()
        if order ==1:
            moment,error = integrate.quad(lambda x: x* self(x), self.domain[0], self.domain[1],epsabs=1e-5,epsrel=1e-5)
            # print(error)
        self_integ = self.Total_Integration()
        # print(moment,error,self_integ)
        if np.abs(self_integ)<1e-12:
            return (self.domain[0]+self.domain[1])/2
        moment = moment/self_integ
        if debug:
            print(moment)
        return moment
    # def Merge_Values(self):
    #     # self.Merge_Domains()
    #     x_array = np.linspace(np.min(self.domains[:,0]),np.max(self.domains[:,1]),101)
    #     vals = np.array([self(x_array[i]) for i in range(x_array.shape[0])])
    #     # print(x_array,vals)
    #     f = interp1d(x_array,vals,kind='cubic')
    #     self.func = f
    
    # def Merge_Domains(self):
    #     # print('called')
    #     # traceback.print_stack()
    #     new_domain = np.array([[np.min(self.domains[:,0]),np.max(self.domains[:,1])]])
    #     self.domains = new_domain
    
    def Convolve(self,conv_func:'XRSA_LinearFunc'):
        selftype = get_type_name(self)
        convolvedtype = get_type_name(conv_func)
        if 'Zero' in selftype or 'Zero' in convolvedtype:
            return XRSA_Zero_Linear_Func()
        # self.Merge_Domains()
        # conv_func.Merge_Domains()
        self_center = (self.domain[1]+self.domain[0])/2
        conv_center = (conv_func.domain[1]+conv_func.domain[0])/2
        common_center = self_center+conv_center
        self_range = self.domain[1]-self.domain[0]
        conv_range = conv_func.domain[1]-conv_func.domain[0]
        common_range = np.maximum(self_range,conv_range)
        ratio = self_range/conv_range
        if ratio>20:
            integ = conv_func.Total_Integration()
            new_func = lambda x: self.func(x-conv_center)*integ
            return XRSA_LinearFunc(new_func,common_center-common_range/2,common_center+common_range/2)
        elif ratio<0.05:
            integ = self.Total_Integration()
            new_func = lambda x: conv_func.func(x-self_center)*integ
            return XRSA_LinearFunc(new_func,common_center-common_range/2,common_center+common_range/2)
        else:
            x_array = np.linspace(-common_range/2,common_range/2,1001)
            self_func_trans = lambda x: self(x+self_center)
            conv_func_trans = lambda x: conv_func(x+conv_center)
            y_self = self_func_trans(x_array)
            y_conv = conv_func_trans(x_array)
            convolved_val = fftconvolve(y_self, y_conv, mode='same')*(x_array[1]-x_array[0])
            convolved_x = np.linspace(common_center-common_range/2,common_center+common_range/2,1001)
            convolved_func = interpolate.interp1d(convolved_x,convolved_val,kind='cubic')
            return XRSA_LinearFunc(convolved_func,common_center-common_range/2,common_center+common_range/2)
    
    def plot_func(self,ax=None):
        show = False
        if ax is None:
            fig=plt.figure()
            ax = fig.add_subplot(111)
            show = True
        # print(self.domains)
        # print(self)
        x = np.linspace(self.domain[0]+1e-12,self.domain[1]-1e-12,1000)
        y = np.array([self(x[j]) for j in range(1000)])
        # print(x,y)
        ax.plot(x,y,color='blue')
        if show:
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            return ax

class XRSA_Wavelength_Func(XRSA_LinearFunc):
    def __init__(self, func, lower_limit=-np.inf, upper_limit=np.inf) -> None:
        super().__init__(func, lower_limit, upper_limit)
        
    def plot_func(self, ax=None):
        show = False
        if ax is None:
            fig=plt.figure()
            ax = fig.add_subplot(111)
            show = True
        # print(self.domains)
        # print(self)
        x = np.linspace(self.domain[0]+1e-12,self.domain[1]-1e-12,1000)
        y = np.array([self(x[j]) for j in range(1000)])
        # print(x,y)
        ax.plot(x,y,color='blue')
        if show:
            plt.xlabel('wavelength(\AA)')
            plt.ylabel('intensity(W/($m^2\cdot rad^2 \cdot \AA$))')
            plt.show()
        else:
            return ax

def Generate_Voigt_Wavelength_Func(intensity,central_wavelength,doppler_sigma,lorentzian_gamma=0):
    func = lambda x: intensity*voigt_profile(x-central_wavelength,doppler_sigma,lorentzian_gamma)
    root1 = fsolve(lambda x: func(x)-1e-2*func(central_wavelength),x0=central_wavelength-doppler_sigma)
    root2 = fsolve(lambda x: func(x)-1e-2*func(central_wavelength),x0=central_wavelength+doppler_sigma)
    return XRSA_Wavelength_Func(func,root1,root2)


class XRSA_Zero_Linear_Func(XRSA_LinearFunc):
    def __init__(self) -> None:
        super().__init__(lambda x: 0)
        
    def __add__(self,new_ob):
        output = copy.deepcopy(new_ob)
        return output
    
    def __mul__(self,new_ob):
        output = XRSA_Zero_Linear_Func()
        return output
    
    def Normalization(self):
        return 0
    
    def Total_Integration(self):
        return 0
    
    def Get_moments(self, _):
        return 0
    
    def plot_func(self, ax=None):
        return None

class XRSA_Zero_Wavelength_Func(XRSA_Zero_Linear_Func):
    def __init__(self) -> None:
        super().__init__()
        
if __name__ == '__main__':
    a = XRSA_Wavelength_Func(2.0,3.95,0.001,0)
    print(a.Total_Integration())
    ax = a.plot_func()
    plt.show()