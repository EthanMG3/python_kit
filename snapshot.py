#=================================
#======= Snapshot Module =========
#-----------content---------------
# >> read: line **-line **
# >> poloishow
# >> poloipsi
# >> poloitheta
# >> profileshow
# >> pdfshow
# >> pdf2dshow
# >> fluxshow
# >> fluxspectrum
# >> poloiRm
#---------------------------------
# --> Yangyang Yu
# --> E-mail: 1701110123@pku.edu.cn
# --> 2019/09/25
#
# --> modified by Handi and Yangyang
#=================================

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.gridspec as gridspec
from scipy.fftpack import fft,ifft

# Definitions, Run first!
def read(fname,kind=0): 
  """
  This reads snap.out files into something that can be read by python more easily.
  kind=0 reads all data
      -> return (profile, pdf, poloidata,fluxdata,pdf2d,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  kind=1 reads just profile
      -> return (profile,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  kind=2 reads just pdf
      -> return (pdf,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  kind=3 reads just poloidal
      -> return (poloidata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  kind=4 reads just flux
      -> return (fluxdata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  kind=5 reads just pdf2d
      -> return (pdf2d,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  """
  try:  
    lines = open(fname, 'r').readlines()
    nspecies=int(lines[0])  # of species: ion, electron, EP, impuries
    nfield=int(lines[1])    # of field variables: phi, a_para, fluidne
    nvgrid=int(lines[2])    # of grids in energy and pitch
    mpsi=int(lines[3])      # of radial grids: mpsi+1
    mtgrid=int(lines[4])    # of poloidal grids
    mtoroidal=int(lines[5]) # of toroidal (parallel) grids
    tmax=float(lines[6])    # upper bound of temperature
    print("nspecies=",nspecies,",","nfield=",nfield,",","nvgrid=",nvgrid,",","mpsi=",mpsi,",","mtgrid=",mtgrid,",","mtoroidal=",mtoroidal,",","tmax=",tmax)

    sect1=mpsi*6*nspecies
    sect2=nvgrid*4*nspecies
    sect3=mtgrid*mpsi*(nfield+2)
    sect4=mtgrid*mtoroidal*nfield
    sect5=nvgrid*nvgrid*2*nspecies
    
    #print(sect1+sect2+sect3+sect4+sect5+7)

    profileseries=np.zeros(sect1,dtype=float)
    pdfseries=np.zeros(sect2,dtype=float)
    poloidataseries=np.zeros(sect3,dtype=float)
    fluxdataseries=np.zeros(sect4,dtype=float)
    pdf2ddataseries=np.zeros(sect5,dtype=float)
    
    profile=np.zeros((mpsi,6,nspecies),dtype=float)
    pdf=np.zeros((nvgrid,4,nspecies),dtype=float)
    poloidata=np.zeros((mtgrid,mpsi,nfield+2),dtype=float)
    fluxdata=np.zeros((mtgrid,mtoroidal,nfield),dtype=float)
    pdf2ddata=np.zeros((nvgrid,nvgrid,2,nspecies),dtype=float)

    ## this separates the files into the four sections as a series of float numbers
    if kind==0 or kind==1:
        for i in range(0,sect1):
            profileseries[i]=float(lines[7+i])
    if kind==0 or kind==2:
        for i in range(0,sect2): 
            pdfseries[i]=float(lines[7+sect1+i])
    if kind==0 or kind==3:
        for i in range(0,sect3):
            poloidataseries[i]=float(lines[7+sect1+sect2+i])
    if kind==0 or kind==4:
        for i in range(0,sect4): 
            fluxdataseries[i]=float(lines[7+sect1+sect2+sect3+i])
    if kind==0 or kind==5:
        for i in range(0,sect5): 
            pdf2ddataseries[i]=float(lines[7+sect1+sect2+sect3+sect4+i])
        
    ## this reorganizes the four sections into the data format we expect in idl
    if kind==0 or kind==1:
        for i in range(0,nspecies):
            for j in range(0,6):
                for k in range(0,mpsi):
                    profile[k,j,i]=profileseries[k+(j*mpsi)+(i*mpsi*6)]
                    
    if kind==0 or kind==2:
        for i in range(0,nspecies):
            for j in range(0,4):
                for k in range(0,nvgrid):
                    pdf[k,j,i]=pdfseries[k+(j*nvgrid)+(i*nvgrid*4)]
                    
    if kind==0 or kind==3:            
        for i in range(0,nfield+2):
            for j in range(0,mpsi):
                for k in range(0,mtgrid):
                    poloidata[k,j,i]=poloidataseries[k+(j*mtgrid)+(i*mpsi*mtgrid)]
    
    if kind==0 or kind==4:    
        for i in range(0,nfield):
            for j in range(0,mtoroidal):
                for k in range(0,mtgrid):
                    fluxdata[k,j,i]=fluxdataseries[k+(j*mtgrid)+(i*mtgrid*mtoroidal)]
    
    if kind==0 or kind==5:    
        for i in range(0,nspecies):
            for j in range(0,2):
                for k in range(0,nvgrid):
                    for l in range(0,nvgrid):
                        pdf2ddata[l,k,j,i]=pdf2ddataseries[l+k*nvgrid+(j*nvgrid*nvgrid)+(i*nvgrid*nvgrid*2)]
                    
    if kind==0:
        return (profile, pdf, poloidata,fluxdata,pdf2ddata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
    elif kind==1:
        return (profile,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
    elif kind==2:
        return (pdf,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
    elif kind==3:
        return (poloidata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
    elif kind==4:
        return (fluxdata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
    elif kind==5:
        return (pdf2ddata,nspecies,nfield,nvgrid,mpsi,mtgrid,mtoroidal,tmax)
  except:
    print("Something went wrong!")
    return

###################################
## plot 
###################################
def poloishow(poloidata, kind = 0, savefig = 0):
    """
    show poloidal cross section
    # kind =
        0: plot phi & apara & dene (default)
        1: phi, 2: apara, 3: dene
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    x = poloidata[:,:,3]
    y = poloidata[:,:,4]
    if kind==0:
        phi = poloidata[:,:,0]
        apara = poloidata[:,:,1]
        dene = poloidata[:,:,2]
        fig, sub = plt.subplots(1,3,figsize=(15,4), dpi = 120)
        figphi = sub[0].contourf(x, y, phi, 40, cmap = 'jet')
        sub[0].set(xlabel='R', ylabel='Z',title='phi')
        figapara = sub[1].contourf(x, y, apara, 40, cmap = 'jet')
        sub[1].set(xlabel='R', ylabel='Z',title='apara')
        figdene = sub[2].contourf(x, y, dene, 40, cmap = 'jet')
        sub[2].set(xlabel='R', ylabel='Z',title='dene')
    elif kind==1:
        phi = poloidata[:,:,0]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figphi = ax.contourf(x,y,phi,40,cmap='jet')
        ax.set(xlabel='R', ylabel='Z',title='phi')
        fig.colorbar(figphi)
    elif kind==2:
        apara = poloidata[:,:,1]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figapara = ax.contourf(x,y,apara,40,cmap='jet')
        ax.set(xlabel='R', ylabel='Z',title='apara')
        fig.colorbar(figapara)
    elif kind==3:
        dene = poloidata[:,:,2]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figdene = ax.contourf(x,y,dene,40,cmap='jet')
        ax.set(xlabel='R', ylabel='Z',title='dene')
        fig.colorbar(figdene)
        
    if savefig:
        fig.savefig("poloidal cross section.png")
    return


def profileshow(profile, kind=0, nspecies=0, savefig=0):
    """
    # kind
        0: density (default)
        1: flow
        2: energy
    # nspecies
        0: ion (default)
        if nhybrid>0 then nspecies+1, electron
        if fload>0 then nspecies+1, fast ion
        if feload>0 then nspecies+1, fast electron
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, sub = plt.subplots(1,2,figsize=(12,4), dpi = 120)
    sub[0].plot(profile[:,kind*2,nspecies])
    sub[1].plot(profile[:,kind*2+1,nspecies])
    if kind==0:
        sub[0].set(xlabel='psi', ylabel='',title='density (full f)')
        sub[1].set(xlabel='psi', ylabel='',title='density (delta f)')
    elif kind==1:
        sub[0].set(xlabel='psi', ylabel='',title='flow (full f)')
        sub[1].set(xlabel='psi', ylabel='',title='flow (delta f)')
    elif kind==2:
        sub[0].set(xlabel='psi', ylabel='',title='energy (full f)')
        sub[1].set(xlabel='psi', ylabel='',title='energy (delta f)')
        
    if savefig:
        fig.savefig("profile.png")
    return

def pdfshow(pdf, tmax, kind=0, nspecies=0, savefig=0):
    """
    # kind
        0: energy (default)
        1: pitch
    # nspecies
        0: ion (default)
        if nhybrid>0 then nspecies+1, electron
        if fload>0 then nspecies+1, fast ion
        if feload>0 then nspecies+1, fast electron
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    y1=pdf[:,kind*2,nspecies]
    y2=pdf[:,kind*2+1,nspecies]
    x=np.arange(0,len(y1))*tmax/len(y1)
    fig, sub = plt.subplots(1,2,figsize=(12,4), dpi = 120)
    sub[0].plot(x[:-1],y1[:-1],'-o',markersize=2.4, color='blue',linewidth=1)
    sub[0].grid()
    sub[1].plot(x[:-1],y2[:-1],'-+',markersize=4, color='red',linewidth=1)
    sub[1].grid()

    if kind==0:
        sub[0].set(xlabel='v_grid', ylabel='',title='energy (full f)')
        sub[1].set(xlabel='v_grid', ylabel='',title='energy (delta f)')
    elif kind==1:
        sub[0].set(xlabel='v_grid', ylabel='',title='pitch (full f)')
        sub[1].set(xlabel='v_grid', ylabel='',title='pitch (delta f)')
        
    if savefig:
        fig.savefig("pdf.png")
    return


def fluxshow(fluxdata, kind = 0, savefig = 0):
    """
    # kind =
        0: plot phi & apara & dene (default)
        1: phi, 2: apara, 3: dene
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    if kind==0:
        phi = fluxdata[:,:,0]
        apara = fluxdata[:,:,1]
        dene = fluxdata[:,:,2]
        fig, sub = plt.subplots(1,3,figsize=(15,4), dpi = 120)
        figphi = sub[0].contourf(phi, 40, cmap = 'jet')
        sub[0].set(xlabel='toroidal', ylabel='tgrid',title='phi')
        figapara = sub[1].contourf(apara, 40, cmap = 'jet')
        sub[1].set(xlabel='toroidal', ylabel='tgrid',title='apara')
        figdene = sub[2].contourf(phi, 40, cmap = 'jet')
        sub[2].set(xlabel='toroidal', ylabel='tgrid',title='dene')
    elif kind==1:
        phi = fluxdata[:,:,0]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figphi = ax.contourf(phi,40,cmap='jet')
        ax.set(xlabel='toroidal', ylabel='tgrid',title='phi')
        fig.colorbar(figphi)
    elif kind==2:
        apara = fluxdata[:,:,1]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figapara = ax.contourf(apara,40,cmap='jet')
        ax.set(xlabel='toroidal', ylabel='tgrid',title='apara')
        fig.colorbar(figapara)
    elif kind==3:
        dene = fluxdata[:,:,2]
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figdene = ax.contourf(dene,40,cmap='jet')
        ax.set(xlabel='toroidal', ylabel='tgrid',title='dene')
        fig.colorbar(figdene)
        
    if savefig:
        fig.savefig("flux.png")
    return

def pdf2dshow(pdf2d, kind = 0, species=0, T=1.0, emax_inv=0.2, lambmax_inv=0.8, savefig = 0):
    """
    # kind =
        0: delta f**2 & full f(default)
        1: full f
        2: delta f**2
    # species
        0: ion (default)
        if nhybrid>0 then nspecies+1, electron
        if fload>0 then nspecies+1, fast ion
        if feload>0 then nspecies+1, fast electron
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    nvgrid=len(pdf2d[:,0,0,0])
    energybin=np.arange(1,nvgrid+1)/nvgrid/emax_inv*T;
    lambdabin=np.arange(1,nvgrid+1)/nvgrid/lambmax_inv;
    if kind==0:
        fig, sub = plt.subplots(1,2,figsize=(10,4), dpi = 120)
        fig1 = sub[0].contourf(energybin[:-1], lambdabin[:-1], pdf2d[:-1,:-1,0,species], 40, cmap = 'jet')
        sub[0].set(xlabel='energy', ylabel='$\lambda=\mu B_a/E$',title='full f')
        fig2 = sub[1].contourf(energybin[:-1], lambdabin[:-1], pdf2d[:-1,:-1,1,species], 40, cmap = 'jet')
        sub[1].set(xlabel='energy', ylabel='$\lambda=\mu B_a/E$',title='$\delta f^2$')
    elif kind==1:
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        fig1 = ax.contourf(energybin[:-1], lambdabin[:-1], pdf2d[:-1,:-1,0,species],40,cmap='jet')
        ax.set(xlabel='energy', ylabel='$\lambda=\mu B_a/E$',title='full f')
        fig.colorbar(fig1)
    elif kind==2:
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        fig2 = ax.contourf(energybin[:-1], lambdabin[:-1], pdf2d[:-1,:-1,1,species],40,cmap='jet')
        ax.set(xlabel='energy', ylabel='$\lambda=\mu B_a/E$',title='$\delta f^2$')
        fig.colorbar(fig2)
        
    if savefig:
        fig.savefig("pdf2d.png")
    return

def fluxspectrum(fluxdata, kind = 1, savefig = 0):
    """
    show flux spectrum
    # kind =
        0: phi, 1: apara, 2: dene
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    mtgrid = len(fluxdata[:,0,0])
    mtoroidal = len(fluxdata[0,:,0])
    
    # poloidal mode
    mmode = int(mtgrid/5)
    y1 = np.zeros(mmode,dtype=float)
    for i in range(0,mtoroidal):
        yy = fft(fluxdata[:,i,kind])
        y1[0] = y1[0]+abs(yy[0])**2
        for j in range(1,mmode):
            y1[j]=y1[j]+abs(yy[j])**2+abs(yy[mtgrid-j])**2
    y1 = np.sqrt(2*y1/mtoroidal)/mtgrid
    
    # parallel mode
    pmode = int(mtoroidal/5)
    y2 = np.zeros(pmode,dtype=float)
    for i in range(0,mtgrid):
        yy = fft(fluxdata[i,:,kind])
        y2[0] = y2[0]+abs(yy[0])**2
        for j in range(1,pmode):
            y2[j]=y2[j]+abs(yy[j])**2+abs(yy[mtoroidal-j])**2
    y2 = np.sqrt(2*y2/mtgrid)/mtoroidal
    
    # plot
    fig, sub = plt.subplots(1,2,figsize=(12,4), dpi = 120)
    sub[0].plot(y1,'-*')
    sub[0].grid()
    sub[1].plot(y2,'-o')
    sub[1].grid()
  
    if kind==0:
        sub[0].set(xlabel='', ylabel='',title='phi flux poloidal spectrum')
        sub[1].set(xlabel='', ylabel='',title='phi flux parallel spectrum')
    elif kind==1:
        sub[0].set(xlabel='', ylabel='',title='apara flux poloidal spectrum')
        sub[1].set(xlabel='', ylabel='',title='apara flux parallel spectrum')
    elif kind==2:
        sub[0].set(xlabel='', ylabel='',title='dene flux poloidal spectrum')
        sub[1].set(xlabel='', ylabel='',title='dene flux parallel spectrum')
        
    if savefig:
        fig.savefig("spectrum.png")
    return
    
def poloipsi(poloidata, mpsi=2, mtgrid=1, kind = 0, savefig = 0):
    """
    show poloidal cross section
    # kind =
        0: phi, 1: apara, 2: dene
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    y1 = poloidata[0,:,kind]
    y2=np.zeros(mpsi,dtype=float)
    for i in range(0,mpsi):
        for j in range(0,mtgrid):
            y2[i]=y2[i]+poloidata[j,i,kind]*poloidata[j,i,kind]
    y2=np.sqrt(y2/mtgrid)
    
    # plot
    fig, sub = plt.subplots(1,2,figsize=(12,4), dpi = 120)
    sub[0].plot(y1,'-o',markersize=2.4, color='blue',linewidth=1)
    sub[0].grid()
    sub[1].plot(y2,'-+',markersize=4, color='red',linewidth=1)
    sub[1].grid()
  
    if kind==0:
        sub[0].set(xlabel='psi', ylabel='', title='Radial profile of phi (theta=0)')
        sub[1].set(xlabel='psi', ylabel='',title='Radial profile of phi (RMS)')
    elif kind==1:
        sub[0].set(xlabel='psi', ylabel='', title='Radial profile of apara (theta=0)')
        sub[1].set(xlabel='psi', ylabel='',title='Radial profile of apara (RMS)')
    elif kind==2:
        sub[0].set(xlabel='psi', ylabel='', title='Radial profile of dene (theta=0)')
        sub[1].set(xlabel='psi', ylabel='',title='Radial profile of dene (RMS)')
        
    if savefig:
        fig.savefig("cut1d_psi.png")
    return


def poloitheta(poloidata, mpsi=2, mtgrid=1, kind = 0, savefig = 0):
    """
    show poloidal cross section
    # kind =
        0: phi, 1: apara, 2: dene
    # savefig
        0: don't save figure (default)
        1: save figure
    """

    y1 = poloidata[:,int(mpsi/2),kind]
    y2=np.zeros(mtgrid,dtype=float)
    for j in range(0,mtgrid):
        for i in range(0,mpsi):
            y2[j]=y2[j]+poloidata[j,i,kind]*poloidata[j,i,kind]
    y2=np.sqrt(y2/mpsi)
    
    # plot
    fig, sub = plt.subplots(1,2,figsize=(12,4), dpi = 120)
    sub[0].plot(y1,'-o',markersize=2.4, color='deepskyblue',linewidth=1)
    sub[0].grid()
    sub[1].plot(y2,'-+',markersize=4, color='limegreen',linewidth=1)
    sub[1].grid()
  
    if kind==0:
        sub[0].set(xlabel='mtheta', ylabel='', title='Poloidal profile of phi (mpsi/2)')
        sub[1].set(xlabel='mtheta', ylabel='',title='Poloidal profile of phi (RMS)')
    elif kind==1:
        sub[0].set(xlabel='mtheta', ylabel='', title='Poloidal profile of apara (mpsi/2)')
        sub[1].set(xlabel='mtheta', ylabel='',title='Poloidal profile of apara (RMS)')
    elif kind==2:
        sub[0].set(xlabel='mtheta', ylabel='', title='Poloidal profile of dene (mpsi/2)')
        sub[1].set(xlabel='mtheta', ylabel='',title='Poloidal profile of dene (RMS)')
        
    if savefig:
        fig.savefig("cut1d_theta.png")
    return
    

def poloiRm(poloidata, kind = 0, Mside=1, savefig = 0):
    """
    show radial mode structure
    # kind =
        0: phi (default), 1: apara, 2: dene
        
    # Mside = 
        1 (default):
        keep the dominating m +- Mside modes;

    # savefig
        0: don't save figure (default)
        1: save figure
    """
    
    x = poloidata[0,:,3]  # mid-plane x position
    y1 = poloidata[:,:,kind]
    mtgrid = len(y1[:,0])
    mpsi = len(y1[0,:])
    y2 = np.zeros((int(mtgrid/2),mpsi),dtype=float)    
    
    for i in range(0,mpsi):
        ytmp = abs(fft(y1[:,i])/mtgrid)
        y2[:,i] = ytmp[0:int(mtgrid/2)]
        y2[1:-1,i] = 2*y2[1:-1,i]

    # find m of the dominating mode
    m = int(np.argmax(y2)/mpsi)
    
    
    # plot
    linestyles=['--','-.','-',':']
    markers=['','*','+','o']
    colors=['b', 'r', 'y', 'r']
    fig, ax = plt.subplots(figsize=(5.2,4), dpi = 120)
    mstart = max(0,m-Mside)
    mend = min(mpsi,m+Mside+1)
    
    for i in range(mstart,mend):
        ax.plot(x,y2[i,:],linestyle=linestyles[i%4],color=colors[i%3],label="m = "+str(i))
  
    if kind==0:
        ax.set(xlabel='R', ylabel='', title='Radial mode structure of phi')
    elif kind==1:
        ax.set(xlabel='R', ylabel='', title='Radial mode structure of Apara')
    elif kind==2:
        ax.set(xlabel='R', ylabel='', title='Radial mode structure of densitye')

    ax.legend()
    ax.grid()
    
    if savefig:
        fig.savefig("R_mode_structure.png")
    return    

