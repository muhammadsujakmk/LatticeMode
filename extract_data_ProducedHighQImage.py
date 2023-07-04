import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from PIL import ImageFont, ImageDraw, Image
import cv2
import matplotlib as mpl
import os

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\\"

def main(): 
    idx = 3 
    if idx==2:
        label = "Reflectance"
    elif idx==3:
        label = "Transmittance"
    else:
        label = "Absorbance"
    Dn = "array_water_TE_smallAOI_0_10" 
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
    x, y, z = np.genfromtxt(filename,unpack=True)
    xi = np.linspace(x.min(), x.max(), 1000)
    yi = np.linspace(y.min(), y.max(), 1000)
    Xi,Yi=np.meshgrid(xi,yi) 
    zi = scipy.interpolate.griddata((x,y), z, (Xi,Yi), method="linear")
    nRIsup = 1.5 
    nRIsub = 1.3325
    m1 = 1
    m2 = 2 
    kx1sub,Erg1sub,kx2sub,Erg2sub = Rayleigh_Anomaly(nRIsub,m1)  
    #kx1m2sub,Erg1m2sub,kx2m2sub,Erg2m2sub = Rayleigh_Anomaly(nRIsub,m2)  
    kx1sup,Erg1sup,kx2sup,Erg2sup = Rayleigh_Anomaly(nRIsup,m1)  
    #kx1m2sup,Erg1m2sup,kx2m2sup,Erg2m2sup = Rayleigh_Anomaly(nRIsup,m2)  
    plt.figure(figsize=(9,7))
    plt.subplots_adjust(left=.1, right=1.05, top=.95, bottom=.11) 
    im = plt.contourf(xi,yi,zi,levels=np.linspace(minn,maxx,200),cmap="binary",extend='neither')
    cbar = plt.colorbar()
    ticks = np.linspace(minn,maxx,num=2)
    formatted_ticks = ["{:.1f}".format(tick) for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(formatted_ticks)
    cbar.ax.tick_params(labelsize=20)
    plt.gca().set_facecolor('black')
    #plt.title('{}'.format(label), fontsize=20)
    #plt.xlabel('k$_{//}$ ($\u03BCm^{-1}$)', fontsize=15)
    plt.xlabel('k$_{//}$ ($nm^{-1}$)', fontsize=20)
    plt.ylabel('Photon Energy (eV)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    lw = 3
    plt.plot(kx1sub,Erg1sub,"--",label="$\u03bb_{rayleigh}^{substrate} (+1)$",linewidth=lw,c="w") 
    plt.plot(kx2sub,Erg2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (-1)$",linewidth=lw,c="k") 
    #plt.plot(kx1m2sub,Erg1m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sub,Erg2m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (-2)$",linewidth=3) 
    plt.plot(kx1sup,Erg1sup,label="$\u03bb_{rayleigh}^{superstrate} (+1)$",linewidth=lw,c="w") 
    plt.plot(kx2sup,Erg2sup,label="$\u03bb_{rayleigh}^{superstrate} (-1)$",linewidth=lw,c="k") 
    #plt.plot(kx1m2sup,Erg1m2sup,label="$\u03bb_{rayleigh}^{superstrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sup,Erg2m2sup,label="$\u03bb_{rayleigh}^{superstrate} (-2)$",linewidth=3) 
    plt.legend(loc='lower right',fontsize=17) 
    #plt.xticks(np.arange(0,2.1,0.5)) 
    plt.xticks(np.arange(0e-3,2.1e-3,0.5e-3)) 
    plt.xlim(x.min(),x.max()) 
    plt.ylim(y.min(),y.max()) 
    #plt.savefig("Px Pol-Wavelength= {}nm.jpg".format(wl))
    plt.show()
    os.remove(filename)

def interPol(x,y):
    return scipy.interpolate.interp1d(x,y,'cubic')

def MergingImage():
    path=r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\Images\FOR PAPER\\"
    img_01 = Image.open(path+"Transmittance_Water_TE.PNG")
    img_02 = Image.open(path+"Transmittance_Water_TM.PNG")
    img_03 = Image.open(path+"Transmittance_Glycerol_TE.PNG")
    img_04 = Image.open(path+"Transmittance_Glycerol_TM.PNG")
    
    img_01_size = img_01.size
    img_02_size = img_02.size
    img_03_size = img_03.size
    img_04_size = img_04.size
    
    new_im = Image.new('RGBA',(2*img_01_size[0],2*img_01_size[1]),(250,250,250))
    new_im.paste(img_01,(0,0))
    new_im.paste(img_02,(img_01_size[0],0))
    new_im.paste(img_03,(0,img_01_size[1]))
    new_im.paste(img_04,(img_01_size[0],img_01_size[1]))
    new_im.save(path+"Merge_Image.png",quality=95) 
    new_im.show() 

def wvl2eV(wvl):
    return 1240/wvl

def eV2wvl(eV):
    return 1240/eV

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


#MergingImage()
main()
#find_max_min()







