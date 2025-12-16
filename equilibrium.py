#=================================
#===== Equilibrium Module ========
#-----------content---------------
# >> read: line **-line **
# >> fieldshow, the same in idl
# >> fieldfft1d, new add
# >> particleshow, the same in idl
#---------------------------------
# --> Yangyang Yu
# --> E-mail: 1701110123@pku.edu.cn
# --> 2019/09/25
#=================================

import math
import numpy as np
import matplotlib.pyplot as plt

def read(fname): 
  """
  This reads equilibrium.out files into something that can be read by python more easily.
  return (pdata,spdata)
  # data structure:
      pdata -> (lsp1, nrplot+1)      # of 1D radial plots and radial points
          0: poloidal flux function
          1: normalized toroidal flux function
          2: minor radius
          3: major radius
          4: Te
          5: -d(ln(Te))/dr
          6: ne
          7: -d(ln(ne))/dr
          8: Ti
          9: -d(ln(Ti))/dr
         10: ni
         11: -d(ln(ni))/dr
         12: Tf
         13: -d(ln(Tf))/dr
         14: nf
         15: -d(ln(nf))/dr
         16: Zeff
         17: toroidal rotation
         18: radial electric field
         19: q
         20: -d(ln(q))/dr
         21: g
         22: p
         23: psi (r)
         24: torpsi (toroidal flux)
         25: rgpsi (radial grid)
         26: psitor (inversion of torpsi)
         27: psirg (inversion of psirg)
         28: error of spline cos
         29: error of spline sin
      spdata -> (lsp2, lst, nplot+2) # of 2D poloidal plots
          0: x
          1: z
          2: b
          3: J
          4: i
          5: zeta2phi
          6: del
  """
  try:  
    lines = open(fname, 'r').readlines()
    nrplot = int(lines[0])  # number of 1D plots
    lsp1 = int(lines[1])    # radial points
    pdata=np.zeros((lsp1, nrplot+1),dtype=float)
    
    for j in range(0,nrplot+1):
        for i in range(0,lsp1):
            pdata[i,j]=float(lines[2+j*lsp1+i])
            
    nplot = int(lines[2+lsp1*(nrplot+1)])
    lsp2 = int(lines[3+lsp1*(nrplot+1)])
    lst = int(lines[4+lsp1*(nrplot+1)])
    spdata=np.zeros((lsp2, lst, nplot+2),dtype=float)
    for k in range(0,nplot+2):
        for j in range(0,lst):
            for i in range(0,lsp2):
                spdata[i,j,k]=float(lines[5+lsp1*(nrplot+1)+i+j*lsp2+k*lsp2*lst])

    return (pdata,spdata)
  except:
    print("Something went wrong!")
    return


def pshow(pdata, kind = 2, savefig = 0):
    """
    show the 1D radial plots and radial points
    # kind
        0: minor -- psi
        1: major radius -- psi
        2: -d(ln(Te))/dr -- r
        3: Te -- psi
        4: -d(ln(ne))/dr -- r
        5: ne -- psi
        6: -d(ln(Ti))/dr -- r
        7: Ti -- psi
        8: -d(ln(ni))/dr -- r
        9: ni -- psi
       10: -d(ln(Tf))/dr -- r
       11: Tf -- psi
       12: -d(ln(nf))/dr -- r
       13: nf -- psi
       14: Zeff -- psi
       15: rotation -- psi
       16: Er -- psi
       17: q -- psi
       18: q -- r
       19: d q/d psi -- psi
       20: r -- psi
       21: g -- psi
       22: g -- r
       23: P -- psi
       24: P -- r
       25: toroidal flux -- psi
       26: toroidal flux -- r
       27: rg -- psi
       28: rg -- r
       29: d torpsi/d psi -- psi 
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    fig, ax= plt.subplots(figsize=(5,3), dpi = 120)
    if kind==0:
        x=pdata[:,0]
        y=pdata[:,2]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='minor radius',title='inverse-ratio from profile data')
    elif kind==1:
        x=pdata[:,0]
        y=pdata[:,3]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='major radius',title='major radius from profile data')
    elif kind==2:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,5]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(Te))/dr')
    elif kind==3:
        x=pdata[:,0]
        y=pdata[:,4]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='Te')
    elif kind==4:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,7]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(ne))/dr')
    elif kind==5:
        x=pdata[:,0]
        y=pdata[:,6]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='ne')
    elif kind==6:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,9]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(Ti))/dr')
    elif kind==7:
        x=pdata[:,0]
        y=pdata[:,8]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='Ti')
    elif kind==8:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,11]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(ni))/dr')
    elif kind==9:
        x=pdata[:,0]
        y=pdata[:,10]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='ni')
    elif kind==10:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,13]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(Tf))/dr')
    elif kind==11:
        x=pdata[:,0]
        y=pdata[:,12]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='Tf')
    elif kind==12:
        x=pdata[:,23]/pdata[-1,23]
        y=pdata[:,15]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='-d(ln(nf))/dr')
    elif kind==13:
        x=pdata[:,0]
        y=pdata[:,14]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='nf')
    elif kind==14:
        x=pdata[:,0]
        y=pdata[:,16]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='Zeff')
    elif kind==15:
        x=pdata[:,0]
        y=pdata[:,17]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='rotation')
    elif kind==16:
        x=pdata[:,0]
        y=pdata[:,18]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='Er')
    elif kind==17:
        x=pdata[:,0]
        y=pdata[:,19]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='q')
    elif kind==18:
        x=pdata[:,23]
        y=pdata[:,19]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='q')
    elif kind==19:
        x=pdata[:,0]
        y=pdata[:,20]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='d q/d psi')
    elif kind==20:
        x=pdata[:,0]
        y=pdata[:,23]/pdata[-1,23]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='r',title='')
    elif kind==21:
        x=pdata[:,0]
        y=pdata[:,21]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='g')
    elif kind==22:
        x=pdata[:,23]
        y=pdata[:,21]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='g')
    elif kind==23:
        x=pdata[:,0]
        y=pdata[:,22]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='P')
    elif kind==24:
        x=pdata[:,23]
        y=pdata[:,22]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='P')
    elif kind==25:
        x=pdata[:,0]
        y=pdata[:,24]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='toroidal flux')
    elif kind==26:
        x=pdata[:,23]
        y=pdata[:,24]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='toroidal flux')
    elif kind==27:
        x=pdata[:,0]
        y=pdata[:,25]
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='rg')
    elif kind==28:
        x=pdata[:,23]
        y=pdata[:,25]
        ax.plot(x,y)
        ax.set(xlabel='r', ylabel='',title='rg') 
    elif kind==29:
        x=pdata[:,0]
        y=x
        lsp1=len(x)
        y[0]=(pdata[1,24]-pdata[0,24])/(pdata[1,0]-pdata[0,0])
        y[lsp1-1]=(pdata[lsp1-1,24]-pdata[lsp1-2,24])/(pdata[lsp1-1,0]-pdata[lsp1-2,0])
        for i in range(1,lsp1-1):
            y[i]=(pdata[i+1,24]-pdata[i-1,24])/(pdata[i+1,0]-pdata[i-1,0])
        ax.plot(x,y)
        ax.set(xlabel='psi', ylabel='',title='d torpsi/d psi')
     
    ax.grid()       
    if savefig:
        fig.savefig("time revolution of radial profile of field")
    return


def spshow(pdata, spdata, kind = 0, savefig = 0):
    """
    show 2D poloidal plots
    # kind
        0: poloidal mesh (default)
        1: B-field
        2: Jacobian
        3: i current
        4: zeta2psi
        5: delB
        6: B-field at outer boundary
        7: Jacobian at outer boundary
        8: error of spline
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    lsp1=len(pdata[:,0])
    lsp2=len(spdata[:,0,0])
    lst=len(spdata[0,:,0])
    fig, ax= plt.subplots(figsize=(6,5), dpi = 120)
    if kind==0:
        ax.axis("equal")
        for i in range(0,lsp2):
            ax.plot(spdata[i,:,0],spdata[i,:,1],'-b', linewidth=0.3)
        for j in range(0,lst):
            ax.plot(spdata[:,j,0],spdata[:,j,1],'-g', linewidth=0.3)
        ax.set(xlabel='x', ylabel='z',title='poloidal mesh')
    elif kind==1:
        ax.axis("equal")
        figfield = ax.contourf(spdata[:,:,0],spdata[:,:,1],spdata[:,:,2],40,cmap='jet')
        ax.set(xlabel='x', ylabel='z',title='B-field')
        fig.colorbar(figfield)  
    elif kind==2:
        ax.axis("equal")
        figfield = ax.contourf(spdata[:,:,0],spdata[:,:,1],spdata[:,:,3],40,cmap='jet')
        ax.set(xlabel='x', ylabel='z',title='Jacobian')
        fig.colorbar(figfield)
    elif kind==3:
        ax.axis("equal")
        figfield = ax.contourf(spdata[:,:,0],spdata[:,:,1],spdata[:,:,4],40,cmap='jet')
        ax.set(xlabel='x', ylabel='z',title='i current')
        fig.colorbar(figfield)
    elif kind==4:
        ax.axis("equal")
        figfield = ax.contourf(spdata[:,:,0],spdata[:,:,1],spdata[:,:,5],40,cmap='jet')
        ax.set(xlabel='x', ylabel='z',title='zeta2psi')
        fig.colorbar(figfield)
    elif kind==5:
        ax.axis("equal")
        figfield = ax.contourf(spdata[:,:,0],spdata[:,:,1],spdata[:,:,6],40,cmap='jet')
        ax.set(xlabel='x', ylabel='z',title='delB')
        fig.colorbar(figfield)
    elif kind==6:
        y=spdata[-1,:,2]
        x=np.linspace(0,lst-1,lst)*2*np.pi/lst
        ax.plot(x,y)
        ax.set(xlabel='theta', ylabel='',title='B-field at outer boundary') 
        ax.grid()
    elif kind==7:
        y1=spdata[-1,:,3]
        x=np.linspace(0,lst-1,lst)*2*np.pi/lst
        ax.plot(x,y1,'-r',label="spdata")
        y2=(pdata[lsp1-1,21]*pdata[lsp1-1,19]+spdata[-1,:,4])/spdata[-1,:,2]**2
        ax.plot(x,y2,'--b',label="(gq+I)/B^2")
        ax.set(xlabel='theta', ylabel='',title='Jacobian at outer boundary') 
        ax.grid()
        ax.legend()
    elif kind==8:
        y1=pdata[:,28]
        y2=pdata[:,29]
        ax.plot(y1,'-r',label="cos")
        ax.plot(y2,'--b',label="sin")
        ax.set(xlabel='', ylabel='',title='error of spline') 
        ax.grid()
        ax.legend()
        
        
    if savefig:
        fig.savefig("equilibrium profile")
    return


