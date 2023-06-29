import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from PIL import Image

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\2D lattice Strucuture\Lattice with Waveguide\\"
def main(): 
    idx = 2 
    idx_HigOrder =21 
    if idx==2:
        label = "Reflectance"
    elif idx==3:
        label = "Transmittance"
    elif idx==4:
        label = "Absorbance"
    if idx_HigOrder < 10: 
        #label_HigOrder = "S{}1_".format(idx-2)
        label_HigOrder = "S{}1".format(idx_HigOrder-1)
    else:
        #label_HigOrder = "S{}_1_".format(idx-2)
        label_HigOrder = "S{}_1".format(idx_HigOrder-1)
    
    Dn = "array_HfO2_100nm_Glycerol_TE_smallAOI_0_10" 
    filename= "Au_{}.txt".format(Dn)
    file_Out= "Au_{}_{}.txt".format(label_HigOrder,Dn)
    file_Out2= "Au_{}_{}.txt".format(label,Dn)
    #writing_file(filename,file_Out,idx)
    #plot2D(file_Out,label)
    Separate_file(filename,idx,label,Dn)
    Separate_file(filename,idx_HigOrder,label_HigOrder,Dn)
    P = np.linspace(3,3,1) 
    for p in P: 
        file1 = np.loadtxt("Au_{}_{}_AOI{}.txt".format(Dn,label,p))
        file2 = np.loadtxt("Au_{}_{}_AOI{}.txt".format(Dn,label_HigOrder,p))
        """ 
        file2 = np.loadtxt("Au_{}_{}350.0.txt".format(Dn,label))
        file3 = np.loadtxt("Au_{}_{}400.0.txt".format(Dn,label))
        file4 = np.loadtxt("Au_{}_{}450.0.txt".format(Dn,label))
        file5 = np.loadtxt("Au_{}_{}500.0.txt".format(Dn,label))
        file6 = np.loadtxt("Au_{}_{}550.0.txt".format(Dn,label))
        file7 = np.loadtxt("Au_{}_{}600.0.txt".format(Dn,label))
        """ 
        
        x = np.linspace(550,1499,1001) 
        y_f1 = interPol(file1[:,0],file1[:,1]) 
        y_f2 = interPol(file2[:,0],file2[:,1]) 
        """ 
        y_f2 = interPol(file2[:,0],file2[:,1]) 
        y_f3 = interPol(file3[:,0],file3[:,1]) 
        y_f4 = interPol(file4[:,0],file4[:,1]) 
        y_f5 = interPol(file5[:,0],file5[:,1]) 
        y_f6 = interPol(file6[:,0],file6[:,1]) 
        y_f7 = interPol(file7[:,0],file7[:,1]) 
        """ 
        fig, ax = plt.subplots()
        ax2 = ax.twinx() 
        le1 = ax.plot(x,y_f1(x),label="{} at {} deg".format(label,p),linewidth=3) 
        le2 = ax2.plot(x,y_f2(x),label="{}".format(label_HigOrder),linewidth=5,color="k") 
        """ 
        plt.plot(x,y_f2(x),label="P = 350 nm",linewidth=5) 
        plt.plot(x,y_f3(x),label="P = 400 nm",linewidth=5) 
        plt.plot(x,y_f4(x),label="P = 450 nm",linewidth=5) 
        plt.plot(x,y_f5(x),label="P = 500 nm",linewidth=5) 
        plt.plot(x,y_f6(x),label="P = 550 nm",linewidth=5) 
        plt.plot(x,y_f7(x),label="P = 600 nm",linewidth=5) 
        """ 
        let= le1+le2
        lets = [l.get_label() for l in let]
        ax.set_xlabel('Wavelength (nm)', fontsize=15)
        ax.set_ylabel('{}'.format(label), fontsize=15)
        ax.legend(let,lets,fontsize=15,loc='upper right') 
        plt.xticks(fontsize=13) 
        plt.yticks(fontsize=13) 
        ax.set_xlim(550,1500) 
        ax.set_xticks(np.arange(550,1500,100)) 
        #plt.savefig("{} P = {}nm.jpg".format(Dn,p)) 
        plt.show()

def Separate_file(filename,idx,label,Dn):
    lat = np.linspace(0,10,11) 
    for w in lat: 
        fileOut = open("Au_{}_{}_AOI{}.txt".format(Dn,label,w),"w")
        with open(filename,"r") as file:
            lines = file.readlines()[5:]
            for line in lines:
                line = line.split()
                if "{:.0f}".format(round(float(line[0]))) == "{:.0f}".format(w):
                    res = "{} {}\n".format(line[1],line[idx])
                    fileOut.write(res)
        fileOut.close()


def writing_file(filename,file_Out,idx):
    with open(filename,"r") as file:
        lines = file.readlines()[5:]
        with open(file_Out,"w") as file:
            for line in lines:
                line=line.split()
                data = "{} {} {}\n".format(line[0],line[1],line[idx])
                file.write(data)

def Rayleigh_Anomaly(RI,p):
    return RI*p 

def plot2D(filename,label):
    x, y, z = np.genfromtxt(filename,unpack=True)
    xi = np.linspace(x.min(), x.max(), 2000)
    yi = np.linspace(y.min(), y.max(), 2000)
    zi = scipy.interpolate.griddata((x,y), z, (xi[None,:], yi[:, None]), method="cubic")
    nsub = 1 
    nsup = 1.5
    RWsup = Rayleigh_Anomaly(nsup,xi)  
    RWsub = Rayleigh_Anomaly(nsub,xi)  
    fig = plt.figure(figsize=(8,6))
    plt.contourf(xi,yi,zi)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.title('{}'.format(label), fontsize=20)
    plt.xlabel('Period (nm)', fontsize=15)
    plt.ylabel('Wavelength (nm)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(xi,RWsup,label="$\u03bb_{rayleigh}^{superstrate} ($\u00b11$)$",linewidth=5) 
    #plt.plot(xi,RWsub,label="$\u03bb_{rayleigh}^{glass} ($\u00b11$)$",linewidth=5) 
    plt.legend(loc='upper left') 
    plt.ylim(550,1500) 
    #plt.savefig("Px Pol-Wavelength= {}nm.jpg".format(wl))
    plt.show()

def interPol(x,y):
    return scipy.interpolate.interp1d(x,y,'cubic')

def MergeImage():
    img_01 = Image.open("Dn0_1_WG_with_ITO_100nm P = 300.0nm.jpg")
    img_02 = Image.open("Dn0_1_WG_with_ITO_100nm P = 350.0nm.jpg")
    img_03 = Image.open("Dn0_1_WG_with_ITO_100nm P = 400.0nm.jpg")
    img_04 = Image.open("Dn0_1_WG_with_ITO_100nm P = 450.0nm.jpg")
    img_05 = Image.open("Dn0_1_WG_with_ITO_100nm P = 500.0nm.jpg")
    img_06 = Image.open("Dn0_1_WG_with_ITO_100nm P = 550.0nm.jpg")
    img_07 = Image.open("Dn0_1_WG_with_ITO_100nm P = 600.0nm.jpg")
    
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
    new_im.save("Merge_Image.png","PNG") 
    new_im.show() 


#MergeImage()
main()








