#=================================
#========Data1d Module============
#-----------content---------------
# >> read: line **-line **
# >> fieldshow, the same in idl
# >> fieldfft1d, new add
# >> particleshow, the same in idl
# >> fft2d
# >> lineIntegral
# >> growthrate2
# >> gammaOmega2
#---------------------------------
# --> Yangyang Yu
# --> E-mail: 1701110123@pku.edu.cn
# --> 2019/09/25
#=================================

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,fft2

def read(fname,nt=0): 
  """
  This reads data1d.out files into something that can be read by python more easily.
  return (data1di,data1df,data1de,field00,fieldrms)
  # data structure:
      data1d* -> (ndstep,mpsi,mpdata1d)
      field** -> (ndstep,mpsi,nfield)
  """
  try:  
    lines = open(fname, 'r').readlines()
    ndstep=int(lines[0])  # time steps
    if nt>0:
        ndstep=nt
    mpsi=int(lines[1])    # radial grids: mpsi+1
    nspecies=int(lines[2])    # species: ion, electron, EP, impurity
    nhybrid=int(lines[3])     # 0: not load electron; >0: load electron
    mpdata1d=int(lines[4])    # the number of fluxes per species: particle number, energy, angular momentum
    nfield=int(lines[5])      # fields: phi, apara, fluidne
    mfdata1d=int(lines[6])    # the number of quantities per field: 00, rms
   
    sect1=mpsi*mpdata1d
    sect2=mpsi*nfield
   
    data1diseries=np.zeros(sect1*ndstep,dtype=float)
    data1dfseries=np.zeros(sect1*ndstep,dtype=float)
    data1deseries=np.zeros(sect1*ndstep,dtype=float)
    field00series=np.zeros(sect2*ndstep,dtype=float)
    fieldrmsseries=np.zeros(sect2*ndstep,dtype=float)
    
    
    data1di=np.zeros((ndstep,mpsi,mpdata1d),dtype=float)
    data1df=np.zeros((ndstep,mpsi,mpdata1d),dtype=float)
    data1de=np.zeros((ndstep,mpsi,mpdata1d),dtype=float)
    field00=np.zeros((ndstep,mpsi,nfield),dtype=float)
    fieldrms=np.zeros((ndstep,mpsi,nfield),dtype=float)
   
    ## this separates the file into a few of sections as series of float numbers
    for i in range(0,ndstep):
        for j in range(0,sect1):
            data1diseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)])
            if nspecies==2:
                if nhybrid==0:
                    data1dfseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1])
                elif nhybrid>0:
                    data1deseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1])
            if nspecies==3:
                data1dfseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1])
                data1deseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1*2])
        
        for j in range(0,sect2):
            field00series[j+i*sect2]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1*nspecies])
            fieldrmsseries[j+i*sect2]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect2+sect1*nspecies])
        
    ## this reorganizes the sections into the data format we expect in idl
    for i in range(0,ndstep):
        for j in range(0,mpdata1d):
            for k in range(0,mpsi):
                data1di[i,k,j]=data1diseries[k+j*mpsi+i*mpdata1d*mpsi]
                if (nspecies==2 and nhybrid==0) or nspecies==3:
                    data1df[i,k,j]=data1dfseries[k+j*mpsi+i*mpdata1d*mpsi]
                if (nspecies==2 and nhybrid>0) or nspecies==3:
                    data1de[i,k,j]=data1deseries[k+j*mpsi+i*mpdata1d*mpsi]
    
    for i in range(0,ndstep):
        for j in range(0,nfield):
            for k in range(0,mpsi):
                field00[i,k,j]=field00series[k+j*mpsi+i*nfield*mpsi]
                fieldrms[i,k,j]=fieldrmsseries[k+j*mpsi+i*nfield*mpsi]
   
    return (data1di,data1df,data1de,field00,fieldrms)
  except:
    print("Something went wrong!")
    return


def fieldshow(fielddata, fieldtype = 0, kind = 1, savefig = 0):
    """
    show the time revolution of radial profile of field
    # fieldtype
        0: phi (default)
        1: apara
        2: fluidne
    # kind
        0: field00 
        1: fieldrms (default)
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    figfield = ax.contourf(fielddata[:,:,fieldtype],40,cmap='jet')
    if kind==0:
        if fieldtype==0:
            ax.set(xlabel='psi', ylabel='dstep',title='phi00')
        elif fieldtype==1:
            ax.set(xlabel='psi', ylabel='dstep',title='Apara00')
        elif fieldtype==2:
            ax.set(xlabel='psi', ylabel='dstep',title='fluidne00')
    elif kind==1:
        if fieldtype==0:
            ax.set(xlabel='psi', ylabel='dstep',title='phi_rms')
        elif fieldtype==1:
            ax.set(xlabel='psi', ylabel='dstep',title='Apara_rms')
        elif fieldtype==2:
            ax.set(xlabel='psi', ylabel='dstep',title='fluidne_rms')
    fig.colorbar(figfield)        
    if savefig:
        fig.savefig("time revolution of radial profile of field")
    return

    

def particleshow(particledata, particletype = 0, kind = 0, savefig = 0):
    """
    show the time revolution of radial profile of particle
    # particletype
        0: number
        1: energy
        2: angular momentum

    # kind
        0: ion 
        1: EP
        2: electron
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    figparticle = ax.contourf(particledata[:,:,particletype],40,cmap='jet')
    if kind==0:
        species = 'ion'
    elif kind==1:
        species = 'EP'
    elif kind==2:
        species = 'electron'

    if particletype==0:
        ax.set(xlabel='psi', ylabel='dstep',title=f'{species} number flux')
    elif particletype==1:
        ax.set(xlabel='psi', ylabel='dstep',title=f'{species} energy flux')
    elif particletype==2:
        ax.set(xlabel='psi', ylabel='dstep',title=f'{species} angular momentum flux')
    elif particletype==3:
        ax.set(xlabel='psi', ylabel='dstep',title=f'{species} fluxmesh')

    if kind == 0 and particletype == 4:
        ax.set(xlabel='psi', ylabel='dstep',title="Er_mesh")
    elif kind > 0 and particletype == 4:
        ax.set(xlabel='psi', ylabel='dstep',title=f'total mesh at {species}')
  
    fig.colorbar(figparticle)        
    if savefig:
        fig.savefig("time revolution of radial profile of particle")
    return



def fft2d(s2, tstep=1, dpsi=1,xmin=0, xmax=5,  ymin=0, ymax=4, colorflag=0, savefig=0):
    """
    fft 2d analysis 
    # s2: 2d data
    # tstep: time step of the first dimension, unit: Omega_p^-1
    # dpsi: space step of the second dimension, k
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    ndstep = len(s2[:,0])
    ndpsi = len(s2[0,:])
    f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
    k = np.linspace(0,int(ndpsi/2)-1,int(ndpsi/2))/ndpsi/dpsi*2*np.pi
    Y = fft2(s2[:,:])
    P2 = abs(Y/ndstep)
    P1 = P2[0:int(ndstep/2),0:int(ndpsi/2)]
    P1[1:-1,1:-1] = 2*P1[1:-1,1:-1]
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    if colorflag==0:
        figfield = ax.contourf(k,f,np.log(P1),40,cmap='jet')
    else:
        figfield = ax.contourf(k,f,np.log(P1),40,cmap='hsv')
    ax.set(xlim=[xmin,xmax],xlabel='k', ylim=[ymin,ymax], ylabel='$\Omega_i$',title='')
    fig.colorbar(figfield)        
    if savefig:
        fig.savefig("fft2d")
    return (k,f,P1)


def lineIntegral(k,f,P1,kmin=0, kmax=1, xmin=0, xmax=5, ymin=0, ymax=10e3,savefig=0):
    """
    line integral from kmin to kmax for 2d fft spectrum
    # k,f,P1: 2d fft spectrum
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    nf = len(f)
    nk = len(k)
    P3=np.zeros(nf,dtype=float)
    print(k[100])
    for i in range(0,nf):
        P3[i]=P1[i,100]
        n=0
        while n < nk:
            if k[n]>kmin and k[n]<kmax:
                P3[i]=P3[i]+P1[i,n]
            n=n+1
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.plot(f,P3)
    ax.set(yscale='log')
    ax.set(xlim=[xmin,xmax],xlabel='$\Omega_i$', ylim=[ymin,ymax],ylabel='value')
    ax.grid()
    if savefig:
        fig.savefig("frequency sperctrum")
    return (f,P3)



def growthrate2(s2,dt,ymin,ymax):
    """
    find growth rate of a 2d data
    """
    s=s2[:,1]
    ls=len(s)
    ls2=len(s2[1,:])
    for i in range(0,ls):
        s[i]=max(s2[i,:])
    
    s=abs(s)+1.0e-40
    list_s = s.tolist()
    py1=max(list_s[:int(ls/4)])
    px1=list_s.index(max(list_s[:int(ls/4)]))*dt
    py2=max(list_s[int(3*ls/4):])
    px2=list_s.index(max(list_s[int(3*ls/4):]))*dt
    gamma=(np.log(py2)-np.log(py1))/(px2-px1)
 #   print('grwothrate=',gamma, '$\Omega_p$')
    
    fig, ax= plt.subplots(figsize=(6,4), dpi = 120)
    x=np.linspace(0,ls-1,ls)*dt
    
    # plot
    style1 = '-'     # line style for line1
    color1 = 'green'  # color for line1
    width1 = 1 # line width for line1
    label1 = 'signal'# label for line1

    style2 = '--'     # line style for line2
    marker2 = '*'
    markersize2=2
    color2 = 'blue'  # color for line2
    width2 = 1 # line width for line2
    label2 = 'growthrate line'# label for line2
    
    line1, = ax.plot(x,s,linestyle=style1,color=color1,linewidth=width1,label=label1)
    line2, = ax.plot([px1,px2],[py1,py2],linestyle=style2,marker=marker2,markersize=markersize2,color=color2,linewidth=width2,label=label2)

    ax.legend()
    ax.set(xlabel='$\Omega_p^{-1}$', ylim=[ymin,ymax], ylabel='rms',yscale = "log",title=r"$\gamma=$"+f"{gamma}") 
    ax.grid()
    plt.show()


    
def gammaOmega2(s2, tstep=1, dpsi=1, ymin=0, ymax=4, savefig=0):
    """
    growthrate 2d analysis  
    # tstep, unit: Omega_p^-1
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    ndstep = int(len(s2[:,0])/2)
    ndpsi = len(s2[0,:])
    f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
    k = np.linspace(0,int(ndpsi/2)-1,int(ndpsi/2))/ndpsi/dpsi*2*np.pi
    Y1 = fft2(s2[0:ndstep,:])
    P12 = abs(Y1/ndstep)
    P11 = P12[0:int(ndstep/2),0:int(ndpsi/2)]
    P11[1:-1,1:-1] = 2*P11[1:-1,1:-1]
    P11=np.log(P11)
    
    Y2 = fft2(fielddata[ndstep+1:,:,fieldtype])
    P22 = abs(Y2/ndstep)
    P21 = P22[0:int(ndstep/2),0:int(ndpsi/2)]
    P21[1:-1,1:-1] = 2*P21[1:-1,1:-1]
    P21=np.log(P21)
    
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    figfield = ax.contourf(k,f,(P21-P11)/ndstep/tstep,40,cmap='jet')
    ax.set(xlabel='k', ylim=[ymin,ymax], ylabel='$\Omega_i$',title='')
    fig.colorbar(figfield)        
    if savefig:
        fig.savefig("growthrate")
    return
