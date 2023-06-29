import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from PIL import Image
import matplotlib as mpl

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\\"


def main(): 
    idx = 2 
    if idx==2:
        label = "Reflectance"
    elif idx==3:
        label = "Transmittance"
    else:
        label = "Absorbance"
    Dn = "array_water_TM_smallAOI_0_10" 
    filename= "Au_{}.txt".format(Dn)
    file_Out= "Au_{}_{}.txt".format(label,Dn)
    file_dispersion= "Au_{}_{}_dispersion.txt".format(label,Dn)
    #writing_file(filename,file_Out,idx)
    #writing_file_dispersion(filename,file_dispersion,idx)
    plot2D(file_dispersion,label)
    Separate_file(filename,idx,label,Dn)
    #MergeImage(Dn)
    AOI=np.linspace(0,10,11)
    c = ["b","c","g","k"]
    fig, ax = plt.subplots() 
    for ang in AOI: 
        file1 = np.loadtxt(f"Au_{Dn}_{label}{ang}.txt")
        x = np.linspace(500,1000,501) 
        y_f1 = interPol(file1[:,0],file1[:,1]) 
        plt.plot(x,y_f1(x),label="$\u03B8={:.0f} \u00B0$".format(ang),linewidth=5) 

    """ 
    AOI=np.linspace(0,50,6)
    for ang in AOI: 
        file1 = np.loadtxt("Au_{}_{}{}.txt".format(Dn,label,ang))
        x = np.linspace(500,1499,1001) 
        y_f1 = interPol(file1[:,0],file1[:,1]) 
        plt.plot(x,y_f1(x),label="$\u03B8={} \u00B0$".format(ang),linewidth=5) 
    """ 
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
    with open(filename,"r") as file:
        lines = file.readlines()[5:]
        with open(file_Out,"w") as file:
            for line in lines:
                line=line.split()
                data = "{} {} {}\n".format(2*np.pi/float(line[1])*np.sin(np.pi/180*float(line[0]))*1e+03,1240/float(line[1]),float(line[idx])*1e+01)
                #data = "{} {} {}\n".format((line[0]),line[1],line[idx])
                file.write(data)

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
    kx1 = 2*np.pi/res2*np.sin(np.pi/180*AOI)*1e+03
    kx2 = 2*np.pi/res3*np.sin(np.pi/180*AOI)*1e+03   
    Erg1 = 1240/res2 
    Erg2 = 1240/res3 
    return kx1,Erg1,kx2,Erg2

def plot2D(filename,label):
    x, y, z = np.genfromtxt(filename,unpack=True)
    xi = np.linspace(x.min(), x.max(), 5000)
    yi = np.linspace(y.min(), y.max(), 5000)
    zi = scipy.interpolate.griddata((x,y), z, (xi[None,:], yi[:, None]), method="cubic")
    nRIsup = 1.5 
    nRIsub = 1.3325
    m1 = 1 
    m2 = 2 
    kx1sub,Erg1sub,kx2sub,Erg2sub = Rayleigh_Anomaly(nRIsub,m1)  
    #kx1m2sub,Erg1m2sub,kx2m2sub,Erg2m2sub = Rayleigh_Anomaly(nRIsub,m2)  
    kx1sup,Erg1sup,kx2sup,Erg2sup = Rayleigh_Anomaly(nRIsup,m1)  
    #kx1m2sup,Erg1m2sup,kx2m2sup,Erg2m2sup = Rayleigh_Anomaly(nRIsup,m2)  
    fig = plt.figure(figsize=(8,6))
    im=plt.contourf(xi,yi,zi,cmap='gray')
    cbar = plt.colorbar(im,format='%.1f')
    #cbar.set_ticks([0, 0.5, 1]) 
    cbar.ax.tick_params(labelsize=15)
    plt.title('{}'.format(label), fontsize=20)
    plt.xlabel('k$_{//}$ ($\u03BCm^{-1}$)', fontsize=15)
    plt.ylabel('Energy (eV)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(kx1sub,Erg1sub,"--",label="$\u03bb_{rayleigh}^{substrate} (+1)$",linewidth=1,c="k") 
    plt.plot(kx2sub,Erg2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (-1)$",linewidth=1,c="w") 
    #plt.plot(kx1m2sub,Erg1m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sub,Erg2m2sub,"--",label="$\u03bb_{rayleigh}^{substrate} (-2)$",linewidth=3) 
    plt.plot(kx1sup,Erg1sup,label="$\u03bb_{rayleigh}^{superstrate} (+1)$",linewidth=1,c="k") 
    plt.plot(kx2sup,Erg2sup,label="$\u03bb_{rayleigh}^{superstrate} (-1)$",linewidth=1,c="w") 
    #plt.plot(kx1m2sup,Erg1m2sup,label="$\u03bb_{rayleigh}^{superstrate} (+2)$",linewidth=3) 
    #plt.plot(kx2m2sup,Erg2m2sup,label="$\u03bb_{rayleigh}^{superstrate} (-2)$",linewidth=3) 
    plt.legend(loc='best',fontsize=13) 
    plt.xticks(np.arange(0,2.1,0.5)) 
    plt.xlim(x.min(),x.max()) 
    plt.ylim(y.min(),y.max()) 
    #plt.savefig("Px Pol-Wavelength= {}nm.jpg".format(wl))
    plt.show()

def interPol(x,y):
    return scipy.interpolate.interp1d(x,y,'cubic')

def MergeImage(Dn):
    img_01 = Image.open("{} P = 300.0nm.jpg".format(Dn))
    img_02 = Image.open("{} P = 350.0nm.jpg".format(Dn))
    img_03 = Image.open("{} P = 400.0nm.jpg".format(Dn))
    img_04 = Image.open("{} P = 450.0nm.jpg".format(Dn))
    img_05 = Image.open("{} P = 500.0nm.jpg".format(Dn))
    img_06 = Image.open("{} P = 550.0nm.jpg".format(Dn))
    img_07 = Image.open("{} P = 600.0nm.jpg".format(Dn))
    
    img_01_size = img_01.size
    img_02_size = img_02.size
    img_03_size = img_03.size
    img_04_size = img_04.size
    img_05_size = img_05.size
    img_06_size = img_06.size
    img_07_size = img_07.size
    
    new_im = Image.new('RGB',(4*img_01_size[0],2*img_01_size[1]))
    new_im.paste(img_01,(0,0))
    new_im.paste(img_02,(img_01_size[0],0))
    new_im.paste(img_03,(2*img_01_size[0],0))
    new_im.paste(img_04,(3*img_01_size[0],0))
    new_im.paste(img_05,(0,img_01_size[1]))
    new_im.paste(img_06,(img_01_size[0],img_01_size[1]))
    new_im.paste(img_07,(2*img_01_size[0],img_01_size[1]))
    new_im.save("Merge_Image_{}.png".format(Dn),"PNG") 
    new_im.show() 

def wvl2eV(wvl):
    return 1240/wvl

def eV2wvl(eV):
    return 1240/eV
def find_max_min():
    max_val = float('-inf')
    min_val = float('+inf')
    with open('Au_Reflectance_array_water_TM_smallAOI_0_10_dispersion.txt','r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            val = float(line[2])
            if val > max_val:
                max_val=val
            elif val<min_val:
                min_val=val
    print('#####################################')
    print(f'Max is {max_val} and Min is {min_val}')
    print('#####################################')



#MergeImage()
main()
#find_max_min()







