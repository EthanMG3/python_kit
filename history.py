#=================================
#======= History Module ==========
#-----------content---------------
# >> read: line **-line **
# >> particleshow
# >> fieldshow
# >> modeshow
# >> fft1d
# >> growthrate
# >> gammaOmega
# >> findpeak
# >> bicoherence
#---------------------------------
# --> Yangyang Yu
# --> E-mail: 1701110123@pku.edu.cn
# --> 2019/09/26
#=================================

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def read(fname, ndstep=0, printstuff='yes'):
    """
    # read history.out
    return (parthist,fieldhist,modehist,ndstep, nspecies, mpdiag, nfield, modes, mfdiag, tstep)
    # data structure:
      parthist -> (ndstep,mpdiag,nspecies)
      fieldhist -> (ndstep,mfdiag,nfield)
      modehist -> (ndstep,2,modes,nfield)
    """
    lines = open(fname, 'r').readlines()
    if(ndstep==0):
        ndstep=int(lines[0])   # of time steps
    nspecies=int(lines[1]) # of species: ion, electron, EP, impuries
    mpdiag=int(lines[2])   # of quantities per species: density, entropy, momentum, energy, fluxes
    nfield=int(lines[3])   # of field variables: phi, a_para, fluidne
    modes=int(lines[4])    # of modes per field: (n,m)
    mfdiag=int(lines[5])   # of quantities per field: rms, single spatial point, zonal components
    tstep=float(lines[6])  # time step size

    if printstuff=='yes':
       print("# tsteps","# species", "# diagnostics", "# fields", "# modes", "# field quants", "tstep size")
       print(ndstep, nspecies, mpdiag, nfield, modes, mfdiag, tstep)
        
    dstep_len = (mpdiag*nspecies)+(mfdiag*nfield)+(modes*nfield*2)
    total_len = dstep_len*ndstep
 #   if (total_len-len(lines[7:])!=0):
 #       print("Not finished history.out, please reduce mstep or increase run time.")
 #       return
    
    hist_len=[(mpdiag*nspecies),(mfdiag*nfield),(modes*nfield*2)]

    partdata=np.zeros((ndstep,hist_len[0]))
    fielddata=np.zeros((ndstep,hist_len[1]))
    modedata=np.zeros((ndstep,hist_len[2]))

    parthist=np.zeros((ndstep,mpdiag,nspecies))
    fieldhist=np.zeros((ndstep,mfdiag,nfield))
    modehist=np.zeros((ndstep,2,modes,nfield))
    
    ## this separates the file into a few of sections as series of float numbers
    for i in range(0,ndstep):
        for j in range(0,hist_len[0]):
            partdata[i,j]=float(lines[7+j+i*dstep_len])
        for j in range(0,hist_len[1]):
            fielddata[i,j]=float(lines[7+j+i*dstep_len+hist_len[0]])
        for j in range(0,hist_len[2]):
            modedata[i,j]=float(lines[7+j+i*dstep_len+hist_len[0]+hist_len[1]])
        
    ## this reorganizes the sections into the data format we expect in idl
    for k in range(0,nspecies):
        for j in range(0,mpdiag):
            parthist[:,j,k]=partdata[:,j+k*mpdiag]
    for k in range(0,nfield):
        for j in range(0,mfdiag):
            fieldhist[:,j,k]=fielddata[:,j+k*mfdiag]
    for k in range(0,nfield):
        for j in range(0,modes):
            for i in range(0,2):
                modehist[:,i,j,k]=modedata[:,i+j*2+k*modes*2]
    
    return (parthist,fieldhist,modehist, ndstep, nspecies, mpdiag, nfield, modes, mfdiag, tstep)


def particleshow(parthist, pdiag=0, species=0, savefig=0):
    """
    # nspecies
        0: ion (default)
        1: EP
        2: electron
    # pdiag
        0: deltaf (default)
        1: deltaf*deltaf
        2: angmon
        3: angmon*deltaf
        4: energy
        5: energy*deltaf
        6: (vdr+vap)*deltaf
        7: (vdr+vap)*angmon*deltaf
        8: (vdr+vap)*energy*deltaf
        9: real(iout)
        10: real(mp)
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.plot(parthist[:,pdiag,species])
    ax.grid()
    if pdiag==0:
        ax.set(xlabel='dstep', ylabel='',title='density deltaf')
    elif pdiag==1:
        ax.set(xlabel='dstep', ylabel='',title='entropy deltaf^2')
    elif pdiag==2:
        ax.set(xlabel='dstep', ylabel='',title='parallel flow u')
    elif pdiag==3:
        ax.set(xlabel='dstep', ylabel='',title='delta_u')
    elif pdiag==4:
        ax.set(xlabel='dstep', ylabel='',title='energy-1.5')
    elif pdiag==5:
        ax.set(xlabel='dstep', ylabel='',title='energy*deltaf')
    elif pdiag==6:
        ax.set(xlabel='dstep', ylabel='',title='particle flux')
    elif pdiag==7:
        ax.set(xlabel='dstep', ylabel='',title='momentum flux')
    elif pdiag==8:
        ax.set(xlabel='dstep', ylabel='',title='energy flux')
    elif pdiag==9:
        ax.set(xlabel='dstep', ylabel='',title='out of bounds ratio')
    elif pdiag==10:
        ax.set(xlabel='dstep', ylabel='',title='real(mp)')
    
    if savefig:
        fig.savefig("particle history.png")
    return parthist[:,pdiag,species]


def fieldshow(fieldhist, fdiag=0, nfield=0, savefig=0):
    """
    # fdiag
        0: (theta=zeta=0) (default)
        1: 00
        2: ZF RMS
        3: RMS
    # nfield
        0: phi
        1: Apara
        2: fluidne
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.plot(fieldhist[:,fdiag,nfield])
    ax.grid()
    if nfield==0:
        if fdiag==0:
            ax.set(xlabel='dstep', ylabel='',title='phi (theta=zeta=0)')
        elif fdiag==1:
            ax.set(xlabel='dstep', ylabel='',title='phi00')
        elif fdiag==2:
            ax.set(xlabel='dstep', ylabel='',title='phi ZF RMS')
        elif fdiag==3:
            ax.set(xlabel='dstep', ylabel='',title='phi RMS')
    elif nfield==1:
        if fdiag==0:
            ax.set(xlabel='dstep', ylabel='',title='Apara (theta=zeta=0)')
        elif fdiag==1:
            ax.set(xlabel='dstep', ylabel='',title='Apara00')
        elif fdiag==2:
            ax.set(xlabel='dstep', ylabel='',title='Apara ZF RMS')
        elif fdiag==3:
            ax.set(xlabel='dstep', ylabel='',title='Apara RMS')
    elif nfield==2:
        if fdiag==0:
            ax.set(xlabel='dstep', ylabel='',title='fluidne (theta=zeta=0)')
        elif fdiag==1:
            ax.set(xlabel='dstep', ylabel='',title='fluidne00')
        elif fdiag==2:
            ax.set(xlabel='dstep', ylabel='',title='fluidne ZF RMS')
        elif fdiag==3:
            ax.set(xlabel='dstep', ylabel='',title='fluidne RMS')
    
    if savefig:
        fig.savefig("field history.png")
    return


def modeshow(modehist, mode_numb=[0], nfield=0, tstep=1, xmin=0, xmax=3, flagfft=1, savefig=0):
    """
    # flagfft
        0: only keep frequency>0
        1: remain negative frequency
    # nfield
        0: phi
        1: Apara
        2: fluidne
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    linestyles=['--','-.','-',':']
    markers=['','*','+','o']
    colors=['b', 'r', 'y', 'r']
    fig, sub = plt.subplots(2,3,figsize=(18,12), dpi = 120)
    for i in mode_numb:
        yr = modehist[:,0,i,nfield]
        yi = modehist[:,1,i,nfield]
        ya = np.sqrt(yr*yr+yi*yi)
        gamma = np.log(ya[-1]/ya[0])/len(ya)
        print('mode'+str(i+1)+'  gamma=',gamma/tstep)
        xpow=np.linspace(0,len(ya)-1,len(ya))
        ndstep = len(yr)
        if flagfft==0:
            Y = fft(yr)
            P2 = abs(Y/ndstep)
            f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
            P1 = P2[0:int(ndstep/2)]
            P1[1:-1] = 2*P1[1:-1]
        else:
            Y = fft(yr+yi*1j)
            P2 = abs(Y/ndstep)
            f = np.linspace(-int(ndstep/2)+1,int(ndstep/2),2*int(ndstep/2))/ndstep/tstep*2*np.pi
            P1 = np.linspace(0,ndstep-1,ndstep)
            for j in range(0,int(ndstep/2)):
                P1[j]=P2[j+int(ndstep/2)]
            for j in range(int(ndstep/2),ndstep):
                P1[j]=P2[j-int(ndstep/2)]
            xmin=-xmax
        ymax = max(P1[:])*1.1
        ymin = 0
        print('mode'+str(i+1)+'  frequency=',f[np.argmax(P1[:])])
        sub[0,0].plot(yr,linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
        sub[0,1].plot(yi,linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
        sub[0,2].plot(ya,linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
        sub[1,0].plot(yr/np.exp(gamma*xpow),linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
        sub[1,1].plot(yi/np.exp(gamma*xpow),linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
        sub[1,2].plot(f, P1, linestyle=linestyles[i%4],color=colors[i%3],label="mode"+str(i+1))
    if nfield==0:
        sub[0,0].set(xlabel='dstep', ylabel='',title='Re(phi)')
        sub[0,1].set(xlabel='dstep', ylabel='',title='Im(phi)')
        sub[0,2].set(xlabel='dstep', ylabel='',title='Abs(phi)',yscale="log")
        sub[1,0].set(xlabel='dstep', ylabel='',title='Re(phi) nomalized by gamma')
        sub[1,1].set(xlabel='dstep', ylabel='',title='Im(phi) nomalized by gamma')
        sub[1,2].set(xlim=[xmin,xmax],xlabel='frequency', ylabel='',title='spectrum')
    elif nfield==1:
        sub[0,0].set(xlabel='dstep', ylabel='',title='Re(Apara)')
        sub[0,1].set(xlabel='dstep', ylabel='',title='Im(Apara)')
        sub[0,2].set(xlabel='dstep', ylabel='',title='Abs(Apara)',yscale="log")
        sub[1,0].set(xlabel='dstep', ylabel='',title='Re(Apara) nomalized by gamma')
        sub[1,1].set(xlabel='dstep', ylabel='',title='Im(Apara) nomalized by gamma')
        sub[1,2].set(xlabel='frequency', xlim=[xmin,xmax],ylim=[ymin,ymax], ylabel='',title='spectrum')
    elif nfield==2:
        sub[0,0].set(xlabel='dstep', ylabel='',title='Re(fluidne)')
        sub[0,1].set(xlabel='dstep', ylabel='',title='Im(fluidne)')
        sub[0,2].set(xlabel='dstep', ylabel='',title='Abs(fluidne)',yscale="log")
        sub[1,0].set(xlabel='dstep', ylabel='',title='Re(fluidne) nomalized by gamma')
        sub[1,1].set(xlabel='dstep', ylabel='',title='Im(fluidne) nomalized by gamma')
        sub[1,2].set(xlabel='frequency', ylim=[ymin,ymax],xlim=[xmin,xmax], ylabel='',title='spectrum')
    
    sub[0,0].legend()
    sub[0,1].legend()
    sub[0,2].legend()
    sub[1,0].legend()
    sub[1,1].legend()
    sub[1,2].legend()
    sub[1,2].grid()
    
    if savefig:
        fig.savefig("mode history.png")
    return

def fft1d(s, tstep=1,xmin=0,xmax=7):
    """
    # s: signal, 1d data
    # tstep: the time step of data
    # xmin, xmax: frequency of plot range
    """
    ndstep = len(s)
    f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
    Y = fft(s)
    P2 = abs(Y/ndstep)
    P1 = P2[0:int(ndstep/2)]
    P1[1:-1] = 2*P1[1:-1]
    ymax=P1[0]
    ymin=P1[0]
    for i in range(0,len(f)):
        if f[i]<xmax and f[i]>xmin and P1[i]>ymax:
            ymax=P1[i]
        elif f[i]<xmax and f[i]>xmin and P1[i]<ymin:
            ymin=P1[i]
    ymax = ymax*1.1
    ymin = ymin*0.9
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.plot(f, P1)
    ax.set(yscale='log')
    ax.set(xlim=[xmin,xmax], xlabel='$\Omega_p$', ylim=[ymin,ymax], ylabel='value',title='')
    ax.grid()
    return (f,P1)

###################################
## find growth rate
###################################
def growthrate(s,dt,ymin,ymax):
    """
    # s: 1d data
    # tstep: the time step of data
    # ymin, ymax: plot range
    """
    ls=len(s)
    s=abs(s)+1.0e-40
    list_s = s.tolist()
    py1=max(list_s[:int(ls/4)])
    px1=list_s.index(max(list_s[:int(ls/4)]))*dt
    py2=max(list_s[int(3*ls/4):])
    px2=list_s.index(max(list_s[int(3*ls/4):]))*dt
    gamma=(np.log(py2)-np.log(py1))/(px2-px1)
 #   print('growthrate=',gamma, '$\Omega_p$')
    
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
    ax.set(xlabel='istep', ylim=[ymin,ymax], ylabel='rms',yscale = "log",title=r"$\gamma=$"+f"{gamma}") 
    ax.grid()
    plt.show()
    
    
###################################
## fft analysis, find growth rate for each frequency
## yang, 12/07/2020
###################################
def gammaOmega(s, tstep=1, xmin=0,xmax=7):
    """
    # tstep, unit: Omega_p^-1
    """
    ndstep = int(len(s)/2)
    f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
    Y1 = fft(s[:ndstep])
    Y2 = fft(s[ndstep+1:])
    
    P12 = abs(Y1/ndstep)
    P11 = P12[0:int(ndstep/2)]
    P11[1:-1] = 2*P11[1:-1]
    
    P22 = abs(Y2/ndstep)
    P21 = P22[0:int(ndstep/2)]
    P21[1:-1] = 2*P21[1:-1]
    
    gamma=(np.log(P21)-np.log(P11))/(tstep*ndstep)
    
    ymax = max(gamma)
    ymin = min(gamma)
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.plot(f, gamma)
    ax.set(xlim=[xmin,xmax], xlabel='$\Omega_p$', ylim=[ymin,ymax], ylabel='growrate',title='')
    ax.grid()
    return (f,gamma)



def findpeak(f,P1,xmin=0, xmax=3,num=5,threshold=0.05):
    """
    find local peak
    # f, P1: frequency spectrum
    # xmin, xmax: the range of frequency
    # num: the width of local peak 
    # threshold: only keep Peak>P1_max*threshold
    """
    ls=len(f)
    y=max(P1)*threshold
    ymax=0
    for i in range(0,ls):
        if f[i]<xmax and f[i]>xmin and P1[i]>ymax:
            ymax=P1[i]
    y=ymax*threshold
    ymax = ymax*1.1
    ymin = 0
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    for i in range(num,ls-num-1):
        if P1[i]>y and P1[i]==max(P1[i-num:i+num+1]):
            print(f[i],P1[i])
            ax.plot(f[i],P1[i],'+b')
    ax.plot(f, P1)
    ax.set(xlim=[xmin,xmax], xlabel='$\Omega_p$', ylim=[ymin,ymax], ylabel='value')
    ax.grid()

    
###################################
## Digital Bispectral Analysis
## Ref: Young C. Kim and Edward J. Powers, 
##      IEEE TRANSACTIONS ON PLASMA SCIENCE, VOL. PS-7, NO. 2, JUNE 1979, 
##      Digital Bispectral Analysis and Its Applications to Nonlinear Wave Interactions
## 07/07/2023
###################################
def bicoherence(fielddata, tstep=1, ndstep=512, nshift=32, xmax=2, ymax=2, colorflag=0, norm=1, savefig=0):
    """
    # fielddata: 1d time series
    # tstep, unit: Omega_p^-1
    # ndstep: length of fft
    # nshift: number of shift
    # colorflag
        0: pcolor, jet (default)
        1: contourf, hsv
    # norm
        0: bispectrum 
        1: bicoherence (default)
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    npart = int((len(fielddata[:])-ndstep)/nshift)+1
    bic = np.zeros((int(ndstep/2),int(ndstep/2)),dtype=float)
    bic1 = np.zeros((int(ndstep/2),int(ndstep/2),npart),dtype=complex)
    bic2 = np.zeros((int(ndstep/2),int(ndstep/2),npart),dtype=float)
    bic3 = np.zeros((int(ndstep/2),int(ndstep/2),npart),dtype=float)
    f = np.linspace(0,int(ndstep/2)-1,int(ndstep/2))/ndstep/tstep*2*np.pi
    for k in range(0,npart):
        Y = fft((fielddata[k*nshift:k*nshift+ndstep]-np.mean(fielddata[k*nshift:k*nshift+ndstep]))*np.hanning(ndstep))
        P1 = Y[1:int(ndstep/2)+1] # exclude f=0 component
        for i in range(0,len(f)):
            for j in range(0,len(f)-i):
                bic1[i,j,k]=P1[i]*P1[j]*np.conjugate(P1[i+j])
                bic2[i,j,k]=abs(P1[i]*P1[j])**2
                bic3[i,j,k]=abs(P1[i+j])**2
                
    if norm==0:
        tmp=np.amax(abs(np.mean(bic1,2)))
        bic=abs(np.mean(bic1,2))/tmp
    else: 
        bic=abs(np.mean(bic1,2))**2/np.maximum(np.mean(bic2,2)*np.mean(bic3,2), 1e-10)
                
    if colorflag==0:
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figfield = ax.contourf(f,f,bic,50,cmap='jet')
        fig.colorbar(figfield) 
    elif colorflag==1:
        fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
        figfield = ax.contourf(f,f,bic,40,cmap='hsv')
        fig.colorbar(figfield) 
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        nf=60;
        fx=f[f<xmax]
        fy=f[f<ymax]
        X, Y = np.meshgrid(fx,fy)
        
        Z = bic[0:len(fx),0:len(fy)]
        ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
        ax.set_zlim([0, 1])
        #ax.contour(X,Y,Z, zdir='z',offset=-2,cmap='rainbow')
        #ax.set_zlim([-2, 1]);

    if norm==0: 
        ax.set(xlabel='f_1', ylabel='f_2',title='bisperctrum')
    else:
        ax.set(xlabel='f_1', ylabel='f_2',title='bicoherence')
        
    ax.set(xlim=[0,xmax], ylim=[0,ymax])

    if savefig:
        fig.savefig("bicoherence")
    return (f,bic)