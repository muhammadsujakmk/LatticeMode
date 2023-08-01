import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import BSpline, make_interp_spline
from scipy.interpolate import CubicSpline
import matplotlib.ticker as ticker

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\\"


def main():
    gly0 = np.genfromtxt(path()+"TopFilling_glycerol_0thOrder.txt")
    gly1R = np.genfromtxt(path()+"TopFilling_glycerol_Ref_1stOrder.txt")
    gly1T = np.genfromtxt(path()+"TopFilling_glycerol_Tran_1stOrder.txt")
    wat0 = np.genfromtxt(path()+"TopFilling_water_0thOrder.txt")
    wat1R = np.genfromtxt(path()+"TopFilling_water_Ref_1stOrder.txt")
    wat1T = np.genfromtxt(path()+"TopFilling_water_Tran_1stOrder.txt")
    
    # Top filling = Water
    wvl1 = wat0[:,0]
    R1 = wat0[:,1]
    T1 = wat0[:,2]

    wvl11 = wat1R[:,0]
    R11y = wat1R[:,1]    #(0,+-1)
    R11x = wat1R[:,2]    #(+-1,0)
    
    wvl12 = wat1T[:,0]
    T12y = wat1T[:,1]    #(0,+-1)
    T12x = wat1T[:,2]    #(+-1,0)

    # Top filling = Glycerol
    wvl2 = gly0[:,0]
    R2 = gly0[:,1]
    T2 = gly0[:,2]

    wvl21 = gly1R[:,0]
    R21y = gly1R[:,1]    #(0,+-1)
    R21x = gly1R[:,2]    #(+-1,0)
    
    wvl22 = gly1T[:,0]
    T22y = gly1T[:,1]    #(0,+-1)
    T22x = gly1T[:,2]    #(+-1,0)
    
    fig = plt.figure()
    plt.subplots_adjust(left=.12, right=.95, top=.98, bottom=.14)
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0)
    x = np.linspace(500,1000,501) 
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = gs.subplots(sharex='col', sharey='row')
    plott(x, interpol(wvl1,R1)(x),ax1,color='blue',label='Reflectance')
    plott(x, interpol(wvl1,T1)(x),ax1,color='gray',label='Transmittance')
    ax1.tick_params(axis='y',labelsize=13) 
    ax1.tick_params(axis='y',direction='in')
    ax1.tick_params(axis='x',direction='in')
    ax1.set_ylim(0,1) 
    ax1.axvline(x=800,color='red',linestyle='dotted')
    ax1.axvline(x=900,color='k',linestyle='dotted')

    plott(x, interpol(wvl2,R2)(x),ax2,color='blue')
    plott(x, interpol(wvl2,T2)(x),ax2,color='gray')
    ax2.axvline(x=883.32,color='red',linestyle='dotted')
    ax2.axvline(x=900,color='k',linestyle='dotted')
    ax2.tick_params(axis='y',direction='inout')
    ax2.tick_params(axis='x',direction='in')
    
    plott(x, interpol(wvl11,R11y)(x),ax3,color='blue',ls=':',label='(0,\u00b11')
    plott(x, interpol(wvl11,R11x)(x),ax3,color='blue',ls='--',label='(\u00b11,0')
    ax3.tick_params(axis='y',labelsize=13) 
    ax3.set_xlim(500,1000) 
    ax3.set_xticks(np.arange(500,1000,100)) 
    ax3.set_ylim(0,.05) 
    ax3.tick_params(axis='y',direction='in')
    ax3.tick_params(axis='x',direction='in')
    ax3.axvline(x=800,color='red',linestyle='dotted')
    
    plott(x, interpol(wvl12,R21y)(x),ax4,color='blue',ls=':')
    plott(x, interpol(wvl12,R21x)(x),ax4,color='blue',ls='--')
    ax4.axvline(x=883.32,color='red',linestyle='dotted')
    ax4.tick_params(axis='y',direction='inout')
    ax4.tick_params(axis='x',direction='in')

    plott(x, interpol(wvl21,T12y)(x),ax5,color='gray',ls=':',label='(0,\u00b11')
    plott(x, interpol(wvl21,T12x)(x),ax5,color='gray',ls='--',label='(\u00b11,0')
    ax5.tick_params(axis='x',labelsize=13,pad=6,direction='in') 
    ax5.tick_params(axis='y',labelsize=13,direction='in') 
    ax5.set_ylim(round(0%1),.18) 
    ax5.set_yticks(np.arange(0,.18,.08)) 
    ax5.axvline(x=900,color='k',linestyle='dotted')

    plott(x, interpol(wvl22,T22y)(x),ax6,color='gray',ls=':')
    plott(x, interpol(wvl22,T22x)(x),ax6,color='gray',ls='--')
    ax6.tick_params(axis='x',labelsize=13,pad=6) 
    ax6.set_xlim(500,1000) 
    ax6.set_ylim(0,.18) 
    ax6.axvline(x=900,color='k',linestyle='dotted')

    
    ax6.tick_params(axis='x',direction='in')
    ax6.tick_params(axis='y',direction='inout')
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Wavelength (nm)',labelpad=10,fontsize=13)
    plt.ylabel('Transmittance',labelpad=15,fontsize=13)
    plt.savefig('Fig 2.jpg',dpi=300) 
    plt.show()

def interpol(x,y):
    #return CubicSpline(x,y)(xx)
    #return make_interp_spline(x,y)(xx)
    return scipy.interpolate.interp1d(x,y,'linear')



def insertText(x,y,text,suf=plt,ha='center',va='center',fontsize=15,color='black'):
    x_pos = x
    y_pos = y
    return suf.text(x_pos,y_pos,text,ha=ha,va=va,fontsize=fontsize,color=color)
            
def plott(x,y,ax=plt,color='k',ls='-',label=''):
    ax.plot(x,y,label=label,color=color,ls=ls)
    if label != '':
        ax.legend(fontsize=8.3,loc='best')


main()



