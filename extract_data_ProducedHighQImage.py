import numpy as np
import math as mth
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib.ticker import FuncFormatter
from PIL import ImageFont, ImageDraw, Image
import matplotlib as mpl
import os

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\\"

def plot1D():
    file=np.loadtxt(path()+'Array_AuDisk_Sensitivity.txt')
    RI = file[:,0]
    label = ['Fano Resonance','LSPR']
    clr = ['k','r']
    fig=plt.figure()
    for i in range(2): 
        T = file[:,i+1]
        x = np.linspace(1.3325,1.4722,11) 
        x_f = interPol(x,T)
        a1,b1=np.polyfit(RI,T,1) 
        plt.scatter(x,x_f(x),s=100,color=clr[i],label=f'{label[i]}') 
        plt.plot(x,a1*x+b1,'--',color=clr[i],label='Linear Fit') 
    plt.text(1.41,826,'$\dfrac{\u0394\u03BB_{res}}{\u0394n}=631.3 nm/RIU$',color='k',fontsize=15)
    plt.text(1.41,806,'$\dfrac{\u0394\u03BB_{res}}{\u0394n}=310.5 nm/RIU$',color='r',fontsize=15)
    plt.text(1.283,900,'(c)',fontsize=20)
    EdgeSizeGraph(left=.16, right=.89, top=.95, bottom=.16)
    plt.tick_params(axis='y',direction='in',left=True,right=False)
    plt.tick_params(axis='x',direction='in',pad=8,bottom=True,top=False)
    plt.ylim(740,900)
    plt.xlim(1.32,1.49)
    plt.legend(loc='best',fontsize=12)
    yy=np.arange(740,920,20)
    xx=np.arange(1.32,1.49,.04)
    format_labelsy = [f'{tick:.0f}' if tick!=0 else '0' for tick in yy] 
    format_labelsx = [f'{tick:.2f}' for tick in xx] 
    plt.yticks(yy,format_labelsy,fontsize=20)
    plt.xticks(xx,format_labelsx,fontsize=20)
    plt.xlabel('Refractive Index (RIU)',fontsize=20)
    plt.ylabel('$\u03BB_{res}$ (nm)',fontsize=20)
    plt.show()

def plotFig5b():
    file0water=np.loadtxt(path()+'Array_AuDisk_Tran_0Water.txt')
    file80water=np.loadtxt(path()+'Array_AuDisk_Tran_20_80Water.txt')
    file100water=np.loadtxt(path()+'Array_AuDisk_Tran_100Water.txt')
    wvl0 = file0water[:,0]
    wvl80 = file80water[:,0]
    wvl100 = file100water[:,0]
    fig=plt.figure()
    x0 = np.linspace(500,1000,501) 
    x_f100 = interPol(wvl100,file100water[:,1])
    plt.plot(x0,x_f100(x0),label='0% of Glycerol',linewidth=5) 
    for i in range(4): 
        T = file80water[:,i+1]
        x = np.linspace(500,1000,501) 
        x_f80 = interPol(wvl80,T)
        plt.plot(x,x_f80(x),label=str(int(i+1)*20)+'% of Glycerol',linewidth=5) 
    x_f0 = interPol(wvl0,file0water[:,1])
    plt.plot(x,x_f0(x),label='100% of Glycerol',linewidth=5) 
    EdgeSizeGraph(left=.14, right=.91, top=.89, bottom=.16)
    plt.tick_params(axis='y',direction='in',left=True,right=False)
    plt.tick_params(axis='x',direction='in',pad=8,bottom=True,top=False)
    plt.ylim(0,2.5)
    plt.xlim(500,1000)
    insertText(520,2.7,"(b)",fontsize=30,color='black')
    #plt.legend(loc='best')
    yy=np.arange(0,2.6,.5)
    format_labels = [f'{tick:.1f}' if tick!=0 else '0' for tick in yy] 
    plt.yticks(yy,format_labels,fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Wavelength (nm)',fontsize=20)
    plt.ylabel('Transmittance',fontsize=20)
    plt.show()
    
def main(): 
    idx = 3 
    if idx==2:
        label = "Reflectance"
    elif idx==3:
        label = "Transmittance"
    else:
        label = "Absorbance"
    Dn = "array_water_TM_smallAOI_0_10" 
    filename= path()+f"Au_{Dn}.txt"
    file_Out= path()+f"Au_{Dn}_{label}.txt"
    file_dispersion= path()+f"Au_{Dn}_{label}_dispersion.txt"
    #writing_file(filename,file_Out,idx)
     
    writing_file_dispersion(filename,file_dispersion,idx)
    maxx, minn = find_max_min(file_dispersion)
    plot2D(file_dispersion,maxx,minn,label)
    #Separate_file(filename,idx,label,Dn)
    #MergeImage(Dn)
    AOI=np.linspace(0,10,11)
    c = ["b","c","g","k"]
    fig, ax = plt.subplots() 
    """ 
    for ang in AOI: 
        file1 = np.loadtxt(f"Au_{Dn}_{label}{ang}.txt")
        x = np.linspace(500,1000,501) 
        y_f1 = interPol(file1[:,0],file1[:,1]) 
        plt.plot(x,y_f1(x),label="$\u03B8={:.0f} \u00B0$".format(ang),linewidth=5) 

    AOI=np.linspace(0,50,6)
    for ang in AOI: 
        file1 = np.loadtxt("Au_{}_{}{}.txt".format(Dn,label,ang))
        x = np.linspace(500,1499,1001) 
        y_f1 = interPol(file1[:,0],file1[:,1]) 
        plt.plot(x,y_f1(x),label="$\u03B8={} \u00B0$".format(ang),linewidth=5) 
    ax2 = ax.secondary_xaxis("top", functions=(wvl2eV,eV2wvl)) 
    ax2.set_xlabel("Energy (eV)",fontsize=15) 
    ax.set_xlabel('Wavelength (nm)', fontsize=15)
    ax.set_ylabel('{}'.format(label), fontsize=15)
    ax.legend(fontsize=8,loc='best') 
    #ax.tick_params(axis='x',which='both',labeltop='on',labelbottom='on') 
    ax.xaxis.set_tick_params(labelsize=13) 
    ax2.xaxis.set_tick_params(labelsize=13) 
    ax.yaxis.set_tick_params(labelsize=13) 
    #plt.yticks(fontsize=13) 
    ax.set_xlim(500,1000) 
    ax.set_xticks(np.arange(500,1100,100)) 
    #plt.savefig("{} P = {}nm.jpg".format(Dn,p)) 
    plt.show()
    """ 


def Separate_file(filename,idx,label,Dn):
    lat = np.linspace(0,10,11) 
    for w in lat: 
        fileOut = open("Au_{}_{}{}.txt".format(Dn,label,w),"w") 
        with open(filename,"r") as file:
            lines = file.readlines()[5:]
            for line in lines:
                line = line.split()
                if "{:.0f}".format(round(float(line[0]))) == "{:.0f}".format(w):
                    res = "{} {}\n".format(line[1],line[idx])
                    fileOut.write(res) 
        fileOut.close()


def writing_file_dispersion(filename,file_Out,idx):
    with open(filename,"r") as fileIn:
        lines = fileIn.readlines()[5:]
        with open(file_Out,"w") as file:
            for line in lines:
                line=line.split()
                data = "{} {} {}\n".format(2*np.pi/float(line[1])*np.sin(np.pi/180*float(line[0])),1240/float(line[1]),float(line[idx]))
                #data = "{} {} {}\n".format((line[0]),line[1],line[idx])
                file.write(data)
    file.close()
def writing_file(filename,file_Out,idx):
    with open(filename,"r") as file:
        lines = file.readlines()[5:]
        with open(file_Out,"w") as file:
            for line in lines:
                line=line.split()
                data = "{} {} {}\n".format(line[0],line[1],line[idx])
                file.write(data)

def Rayleigh_Anomaly(n,m):
    AOI = np.linspace(0,10,11) 
    p = 600 
    res1 = p/m 
    res2 = (n+np.sin(AOI*np.pi/180))*res1
    res3 = (n-np.sin(AOI*np.pi/180))*res1
    kx1 = 2*np.pi/res2*np.sin(np.pi/180*AOI)
    kx2 = 2*np.pi/res3*np.sin(np.pi/180*AOI)
    Erg1 = 1240/res2 
    Erg2 = 1240/res3 
    return kx1,Erg1,kx2,Erg2

def plot2D(filename,maxx,minn,label):
    # Loading data files 
    x, y, z = np.genfromtxt(filename,unpack=True)
    xi = np.linspace(x.min(), x.max(), 1000)
    yi = np.linspace(y.min(), y.max(), 1000)
    Xi,Yi=np.meshgrid(xi,yi) 
    zi = scipy.interpolate.griddata((x,y), z, (Xi,Yi), method="linear")
    nRIsub = 1.5 
    nRIsup = 1.3325
    m1 = 1
    m2 = 2 
    kx1sub,Erg1sub,kx2sub,Erg2sub = Rayleigh_Anomaly(nRIsub,m1)  
    #kx1m2sub,Erg1m2sub,kx2m2sub,Erg2m2sub = Rayleigh_Anomaly(nRIsub,m2)  
    kx1sup,Erg1sup,kx2sup,Erg2sup = Rayleigh_Anomaly(nRIsup,m1)  
    #kx1m2sup,Erg1m2sup,kx2m2sup,Erg2m2sup = Rayleigh_Anomaly(nRIsup,m2)  
    
    #Plot window setting
    fig=plt.figure(figsize=(8,6))
    ax = fig.subplots() 
    
    #plt.subplots_adjust(left=.02, right=.98, top=.98, bottom=.14) 
    
    ##### for water ####### 
    EdgeSize2D() 
   
    #Plot 2D
    im=plt.contourf(xi,yi,zi,levels=np.linspace(minn,maxx,200),cmap="binary",extend='neither')
    #plt.title('{}'.format(label), fontsize=20)
    #plt.xlabel('k$_{//}$ ($\u03BCm^{-1}$)',labelpad=1, fontsize=20)
    #plt.xlabel('k$_{//}$ ($nm^{-1}$)', fontsize=20)
    #plt.ylabel('Photon Energy (eV)', fontsize=20)
    #plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='y',direction='inout',pad=6,left=True,right=False,length=10)
    plt.tick_params(axis='x',direction='inout',pad=7,bottom=True,top=False,length=10)
    plt.gca().set_facecolor('black')
    
    #Setting colorbar
    cbar = plt.colorbar(im,cax=ax.inset_axes((.88, .05, .03, .3)))
    ticks = np.linspace(minn,maxx,num=2)
    formatted_ticks = ["{:.1f}".format(tick) for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(formatted_ticks)
    cbar.ax.tick_params(labelcolor='white',labelsize=20)
    
    #Plot Rayleigh line
    lw = 3
    plt.plot(kx1sub,Erg1sub,label="$\u03bb_{rayleigh}^{substrate}$",linewidth=lw,c="w") 
    insertText(.001,1.31,"(+1,0)",color='white')
    plt.plot(kx2sub,Erg2sub,linewidth=lw,c="w") 
    insertText(.001,1.58,"(-1,0)",color='white')
    #plt.plot(kx1m2sub,Erg1m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sub,Erg2m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (-2)$",linewidth=3) 
    plt.plot(kx1sup,Erg1sup,"--",label="$\u03bb_{rayleigh}^{superstrate}$",linewidth=lw,c="k") 
    insertText(.001,1.45,"(+1,0)",color='black')
    plt.plot(kx2sup,Erg2sup,"--",linewidth=lw,c="k") 
    insertText(.001,1.77,"(-1,0)",color='black')
    #plt.plot(kx1m2sup,Erg1m2sup,label="$\u03bb_{rayleigh}^{superstrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sup,Erg2m2sup,label="$\u03bb_{rayleigh}^{superstrate} (-2)$",linewidth=3) 
    #plt.legend(loc='lower right',fontsize=25)
    plt.xlim(x.min(),x.max()) 
    plt.ylim(y.min(),y.max()) 
   
    #Labelling the graph
    insertText(.0001,2.4,"(b)",fontsize=30,color='white')
    
    #Tuning x-absis range values
    #plt.xticks(np.arange(0,2.1,0.5)) 
    #plt.xticks(np.arange(0e-3,2.1e-3,0.5e-3)) 
    
    """
    xx=np.arange(0,2.1e-3,0.5e-3)
    format_labels = [f'{tick*1000:.1f}' if tick!=0 else '0' for tick in xx] 
    plt.xticks(xx,format_labels,fontsize=22)
    
    #Removing tick on x and Y
    #ax.yaxis.set_major_locator(plt.NullLocator())
    #ax.xaxis.set_major_locator(plt.NullLocator())
    ticksy, _=plt.yticks()
    plt.yticks(ticksy,[])
    """
    xx=np.arange(0,2.1e-3,0.5e-3)
    format_labels = [f'{tick*1000:.1f}' if tick!=0 else '0' for tick in xx] 
    plt.xticks(xx,format_labels,fontsize=22)
    ticksx, _=plt.xticks()
    plt.xticks(ticksx,[])
    #Showing figure
    plt.show()
    imPath=r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\Images\FOR PAPER\\"
    #plt.savefig(imPath+"Transmittance_Water_TM.png",dpi=300)
    os.remove(filename)

def interPol(x,y):
    return scipy.interpolate.interp1d(x,y,'cubic')

def MergingImage():
    path=r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\Images\FOR PAPER\\"
    #img_01 = cairosvg.svg2png(url=path+"Tran_Water_0th Order.svg",write_to=path+"Tran_Water_0th Order.PNG")
    img_01 = Image.open(path+"Tran_Water_0th Order.PNG")
    img_02 = Image.open(path+"Transmittance_Water_TE.PNG")
    img_03 = Image.open(path+"Transmittance_Water_TM.PNG")
    #img_04 = cairosvg.svg2png(url=path+"Tran_Glycerol_0th Order.svg",write_to=path+"Tran_Glycerol_0th Order.PNG")
    img_04 = Image.open(path+"Tran_Glycerol_0th Order.PNG")
    img_05 = Image.open(path+"Transmittance_Glycerol_TM.PNG")
    img_06 = Image.open(path+"Transmittance_Glycerol_TM.PNG")
    
    img_01_size = img_01.size
    img_02_size = img_02.size
    img_03_size = img_03.size
    img_04_size = img_04.size
    img_05_size = img_05.size
    img_06_size = img_06.size
    
    new_im = Image.new('RGBA',(2*img_01_size[0],3*img_01_size[1]),(250,250,250))
    new_im.paste(img_01,(0,0))
    new_im.paste(img_02,(img_01_size[0],0))
    new_im.paste(img_03,(0,img_01_size[1]))
    new_im.paste(img_04,(img_01_size[0],img_01_size[1]))
    new_im.save(path+"Merge_Image.png",quality=300) 
    new_im.show() 

def wvl2eV(wvl):
    return 1240/wvl

def eV2wvl(eV):
    return 1240/eV

def EdgeSizeGraph(left=.25, right=.92, top=.98, bottom=.14):
    return plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom) 

def EdgeSize2D(left=.02, right=.98, top=.98, bottom=.14):
    return plt.subplots_adjust(left=.02, right=.98, top=.98, bottom=.14) 

def PlotFig4():
    fig=plt.figure(figsize=(4,6))
    y = np.linspace(1240/500,1240/1500,1001) 
    x_f1 = interPol(wvl,T) 
    EdgeSizeGraph()
    plt.plot(x_f1(y),y,linewidth=5) 
    plt.tick_params(axis='y',direction='inout',pad=6,left=True,right=True,length=10)
    plt.tick_params(axis='x',direction='inout',pad=8,bottom=True,top=True,length=10)
    

    plt.ylim(1240/1500,1240/500)
    plt.xlim(1,.26)
    xx=np.arange(1,.24,-0.25)
    format_labels = [f'{tick:.1f}' if tick!=1 else '1' for tick in xx] 
    plt.xticks(xx,format_labels,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Transmittance',fontsize=20)
    plt.ylabel('Photon Energy (eV)',fontsize=20)

def insertText(x,y,text,suf=plt,ha='center',va='center',fontsize=20,color='black'):
    x_pos = x
    y_pos = y
    return suf.text(x_pos,y_pos,text,ha=ha,va=va,fontsize=fontsize,color=color)
def find_max_min(file):
    max_val = float('-inf')
    min_val = float('+inf')
    with open(file,'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            val = float(line[2])
            if val > max_val:
                max_val=val
            elif val<min_val:
                min_val=val
    return max_val, min_val
    """
    print('#####################################')
    print(f'Max is {max_val} and Min is {min_val}')
    print('#####################################')
    """

#plotInset()
#MergingImage()
#plot1D()
plotFig5b()
#main()
#find_max_min()







