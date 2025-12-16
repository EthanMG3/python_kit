#=================================
#========Tracking Module============
#>> read output files generated
#>> from tracking.F90
#-----------content---------------
# >> read1:
# >> read2:
# >> show2:
#---------------------------------
# --> Yangyang Yu
# --> E-mail: yangyy3@uci.edu, yangyangyu0921@gmail.com
# --> 04/29/2024
#=================================

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.gridspec as gridspec
import glob
import plotly.graph_objects as go

# Definitions, Run first!
def read1(directory,pattern='TRACKI.*'): 
    """
    (out1,nparticles)=read1(directory,'TRACKI.*',0)
    # read1: track_particles=1 -> Track a set of particles (or all in poincare)
    # return(out1, nparticles)
    # out1[part tracking number, time, i]
    !> For every particle tracked we write the following quantities i
    !> 0: part tracking number
    !> 1: psi
    !> 2: theta (booz poloidal angle)
    !> 3: zeta  (booz toroidal angle)
    !> 4: E
    !> 5: mu
    !> 6: Pzeta
    !> 7: weight
    !> 8: time
    """
    search_pattern = f"{directory}/{pattern}"
    files = glob.glob(search_pattern)
    sorted_files = sorted(files)
    
    # List to store data from the specific line of all files
    result_arrays = []
    
    for file_path in sorted_files:
        try:
            with open(file_path, 'r') as file:
                content=file.readlines()
                for data_string in content:
                    # Split the string into a list of substrings
                    data_list = data_string.split()

                    # Convert each substring to a floating-point number
                    data_array = [float(item) for item in data_list]

                    # Append the 2D array to the result list
                    result_arrays.append(data_array)
                #print(file_path)
                trackp = np.array(result_arrays)
        except IOError as e:
            print(f"Error opening or reading file {file_path}: {e}")
    
    nparticles=np.count_nonzero(trackp[:,-1] == min(trackp[:,-1])) 
    ndstep=int(len(trackp[:,0])/nparticles)
    #print(nparticles,ndstep)
        
    # Sort by tracked particle number
    sorted_trackp = trackp[trackp[:, 0].argsort()]
    out=np.zeros((nparticles,ndstep,9),dtype=float)
    
    # Sort by time step number
    out1=np.zeros((nparticles,ndstep,9),dtype=float)
    for i in range(0,nparticles):
        out[i,:,:]=sorted_trackp[i*ndstep:(i+1)*ndstep,:]
        out1[i]=out[i][out[i][:, -1].argsort()]
    
    # Make theta and zeta change continuously for frequency analysis
    pi2=2*np.pi
    for i in range(0,nparticles):
        theta=out1[i,:,2]
        zeta=out1[i,:,3]
        iz=0
        it=0
        for j in range(1,ndstep):
            if theta[j-1]-pi2*it<0.1  and pi2-theta[j]<0.1:
                it=it-1
            elif pi2-theta[j-1]+pi2*it<0.1  and theta[j]<0.1:
                it=it+1
            if zeta[j-1]-pi2*iz<0.1  and pi2-zeta[j]<0.1:
                iz=iz-1
            elif pi2-zeta[j-1]+pi2*iz<0.1  and zeta[j]<0.1:
                iz=iz+1
            zeta[j]=zeta[j]+pi2*iz
            theta[j]=theta[j]+pi2*it
        out1[i,:,2]=theta
        out1[i,:,3]=zeta

    return(out1,nparticles)

# Definitions, Run first!
def read2(directory,pattern='TRACKI.*',ndstep=0): 
    """
    (out1,nparticles)=read2(directory,'TRACKI.*',0)
    # read2: track_particles=2 -> Track a set of particles (or all in poincare)
    !> 0: time
    !> 1: psi
    !> 2: theta (booz poloidal angle)
    !> 3: zeta  (booz toroidal angle)
    !> 4: part tracking number
    !> 5: E
    !> 6: mu
    !> 7: Pzeta
    !> 8: q
    !> 9: bfield
    !> 10: X coordinate
    !> 11: Y coordinate
    !> 12: Z coordinate
    !> 13: Grad-B drift (in the binormal direction)
    !> 14: B curvature drift (in the binormal direction)

    # return(trackp)
    """
    search_pattern = f"{directory}/{pattern}"
    files = glob.glob(search_pattern)
    sorted_files = sorted(files)
    
    # List to store data from the specific line of all files
    result_arrays = []
    
    for file_path in sorted_files:
        try:
            with open(file_path, 'r') as file:
                content=file.readlines()
                for data_string in content:
                    # Split the string into a list of substrings
                    data_list = data_string.split()

                    # Convert each substring to a floating-point number
                    data_array = [float(item) for item in data_list]

                    # Append the 2D array to the result list
                    result_arrays.append(data_array)
                #print(file_path)
                trackp = np.array(result_arrays)
        except IOError as e:
            print(f"Error opening or reading file {file_path}: {e}")
    
    nparticles=np.count_nonzero(trackp[:,0] == min(trackp[:,0]))
    ndstep=int(len(trackp[:,0])/nparticles)
    #print(nparticles,ndstep)
        
    # Sort by tracked particle number
    sorted_trackp = trackp[trackp[:, 4].argsort()]
    out=np.zeros((nparticles,ndstep,15),dtype=float)
    # Sort by time step number
    out1=np.zeros((nparticles,ndstep,15),dtype=float)
    for i in range(0,nparticles):
        out[i,:,:]=sorted_trackp[i*ndstep:(i+1)*ndstep,:]
        out1[i]=out[i][out[i][:, 0].argsort()]
    
    # Make theta and zeta change continuously for frequency analysis
    pi2=2*np.pi
    for i in range(0,nparticles):
        theta=out1[i,:,2]
        zeta=out1[i,:,3]
        iz=0
        it=0
        for j in range(1,ndstep):
            if theta[j-1]-pi2*it<0.1  and pi2-theta[j]<0.1:
                it=it-1
            elif pi2-theta[j-1]+pi2*it<0.1  and theta[j]<0.1:
                it=it+1
            if zeta[j-1]-pi2*iz<0.1  and pi2-zeta[j]<0.1:
                iz=iz-1
            elif pi2-zeta[j-1]+pi2*iz<0.1  and zeta[j]<0.1:
                iz=iz+1
            zeta[j]=zeta[j]+pi2*iz
            theta[j]=theta[j]+pi2*it
        out1[i,:,2]=theta
        out1[i,:,3]=zeta

    return(out1,nparticles)

def show2(outp,nparticles=0,nstart=0,orbit3d=0):
    """
    show particle orbits using read2
    # nparticles: the number of orbits
    # nstart: the started number of tracked particles
    # orbit3d
        0: no 3d_orbit
        1: plot 3d_orbit
    """
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    if(nparticles==0):
        nparticles=len(outp[:,0,0])
    for i in range(nstart,nstart+nparticles):
        ax.plot(np.sqrt(outp[i,:,10]**2+outp[i,:,11]**2)-1,outp[i,:,12])
    
    ax.set(xlabel='R/$R_0$-1', ylabel='Z/$R_0$',title='Projection of particle orbits on the poloidal plane')
    ax.axis('equal')
    fig.show()
    
    if(orbit3d>0):
        fig2 = go.Figure()
        for i in range(nstart,nstart+nparticles):
            x = outp[i,:,10]
            y = outp[i,:,11]
            z = outp[i,:,12]

            fig2.add_trace(go.Scatter3d(x=x, y=y, z=z,
                           mode='lines',
                           line=dict(width=5)))

        fig2.update_layout(title='3D particle orbits',
                      scene=dict(
                      xaxis_title='X Axis',
                      yaxis_title='Y Axis',
                      zaxis_title='Z Axis'
                      ),
                      width=800, 
                      height=600,
                      margin=dict(l=0, r=0, b=0, t=50))
        fig2.show()
    
    
def show_until_lost(outp,nparticles=0,nstart=0,orbit3d=0,psiout = 0.1):
    """
    show particle orbits using read2
    # outp: the output of read2
    # nparticles: the number of orbits
    # nstart: the started number of tracked particles
    # psiout: the psi value to stop the tracking
    # orbit3d
        0: no 3d_orbit
        1: plot 3d_orbit
    """
    fig, ax= plt.subplots(figsize=(5.2,4), dpi = 120)
    steps = len(outp[0,:,0])
    if(nparticles==0):
        nparticles=len(outp[:,0,0])
    for i in range(nstart,nstart+nparticles):
        for j in range(steps):
            if (outp[i,j,1]>psiout):
                break
        r = np.sqrt(outp[i,:j,10]**2+outp[i,:j,11]**2)
        alpha = 2*outp[i,:j,2]-10*outp[i,:j,3]
        psi = outp[i,:j,1]/psiout
        ax.plot(np.sqrt(psi)*np.cos(alpha),np.sqrt(psi)*np.sin(alpha))
    
    ax.set(xlabel= '$\\alpha= \Theta - 5* \zeta$', ylabel='s = $\sqrt{\psi_p/\psi_w}$',title='Projection of particle orbits on rotating poloidal plane')
    ax.axis('equal')
    fig.show()
    
    if(orbit3d>0):
        fig2 = go.Figure()
        for i in range(nstart,nstart+nparticles):
            if (np.abs(outp[i,0,6]/outp[i,0,5]-outp[i,-1,6]/outp[i,-1,5])>0.05):
                for j in range(steps):
                    if (outp[i,j,1]>psiout):
                        break
                x = outp[i,:j,10]
                y = outp[i,:j,11]
                z = outp[i,:j,12]
                color = (2*outp[i,0,2]+10*outp[i,0,3])%(2*np.pi)
                color = color/(2*np.pi)
        
                fig2.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                mode='markers',
                                marker=dict(size=2,
                                opacity=color)))

        fig2.update_layout(title='3D particle orbits',
                      scene=dict(
                      xaxis_title='X Axis',
                      yaxis_title='Y Axis',
                      zaxis_title='Z Axis'
                      ),
                      width=800, 
                      height=600,
                      margin=dict(l=0, r=0, b=0, t=50))
        fig2.show()
    return fig2
        
    

def show_until_lost_c(outps,nparticles=0,nstart=0,orbit3d=0,psiout = 0.1,labels = None,max_steps=None,plot1d=False,legend = 1,lw = 1):
    """
    show multiple particle orbits using read2
    # outps: the output of read2
    # nparticles: the number of orbits
    # nstart: the started number of tracked particles
    # psiout: the psi value to stop the tracking
    # labels: the labels of the orbits
    # max_steps: the maximum number of steps to track
    # plot1d: plot 1d orbit
    # legend: plot legend
    # lw: line width
    # orbit3d
        0: no 3d_orbit
        1: plot 3d_orbit
    """
    fig, ax= plt.subplots(figsize=(7,7))
    fontsize = 40
    
    dotsize = 150
    if labels is None:
        labels = [None]*len(outps)
    #loop over runs to compare
    for p in range(len(outps)):
        steps = len(outps[p][0,:,0])
        if max_steps is None:
            max_steps = steps
        if(nparticles==0):
            nparticles=len(outps[p][:,0,0])
        for i in range(nstart,nstart+nparticles):
            for j in range(max_steps):
                if (outps[p][i,j,1]>psiout):
                    break
            
            chi = 2*outps[p][i,:j,2]-10*outps[p][i,:j,3]
            psi = outps[p][i,:j,1]/psiout
            line, = ax.plot(np.sqrt(outps[p][i,:j,10]**2+outps[p][i,:j,11]**2)-1,outps[p][i,:j,12],label = labels[p],lw = lw,zorder =1)
            if j != max_steps-1:
                ax.scatter(np.sqrt(outps[p][i,j-1,10]**2+outps[p][i,j-1,11]**2)-1,outps[p][i,j-1,12],marker = 'x',color = line.get_color(),s = dotsize+50,linewidth = lw+2,zorder =2)
            if p ==0:
                ax.scatter(np.sqrt(outps[p][i,0,10]**2+outps[p][i,0,11]**2)-1,outps[p][i,0,12],marker = 'o',color = 'black',label = 'start point',s = dotsize,zorder = 2)
    ax.set_xlabel('X $(R_0)$', fontsize = fontsize)
    ax.set_ylabel('Z $(R_0)$',fontsize = fontsize)
    ax.tick_params(axis='both',labelsize = fontsize)
    ax.axis('equal')
    if legend==1:
        ax.legend(fontsize = fontsize)
    fig.show()
    
    #helical 
    fig, ax= plt.subplots(figsize=(7,7))
    if labels is None:
        labels = [None]*len(outps)
    #loop over runs to compare
    for p in range(len(outps)):
        steps = len(outps[p][0,:,0])
        if max_steps is None:
            max_steps = steps
        if(nparticles==0):
            nparticles=len(outps[p][:,0,0])
        for i in range(nstart,nstart+nparticles):
            for j in range(max_steps):
                if (outps[p][i,j,1]>psiout):
                    break
            
            chi = 2*outps[p][i,:j,2]-10*outps[p][i,:j,3]
            psi = outps[p][i,:j,1]/psiout

            line, = ax.plot(np.sqrt(psi)*np.cos(chi),np.sqrt(psi)*np.sin(chi),label = labels[p],lw = lw,zorder = 0)
            if j != max_steps-1:
                ax.scatter(np.sqrt(psi[-1])*np.cos(chi[-1]),np.sqrt(psi[-1])*np.sin(chi[-1]),marker = 'x',color = line.get_color(),s =dotsize+50,linewidth = lw+2,zorder = 2)
            if p ==0:
                ax.scatter(np.sqrt(psi[0])*np.cos(chi[0]),np.sqrt(psi[0])*np.sin(chi[0]),marker = 'o',color = 'black',label = 'start point',s=dotsize,zorder = 2)
    th = np.linspace(0,2*np.pi,200)
    ax.plot(0.25*np.cos(th),0.25*np.sin(th),'--',color = 'black')
    ax.plot(0.5*np.cos(th),0.5*np.sin(th),'--',color = 'black')
    ax.plot(0.75*np.cos(th),0.75*np.sin(th),'--',color = 'black')
    ax.plot(np.cos(th),np.sin(th),'--',color = 'black')
    ax.set_xlabel('$\sqrt{\psi} cos(\\chi$)',fontsize = fontsize)
    ax.set_ylabel('$\sqrt{\psi} sin(\\chi)$',fontsize = fontsize)
    ax.axis('equal')
    ax.tick_params(axis='both',labelsize = fontsize)
    if legend==2:
        ax.legend(fontsize = fontsize)
    fig.show()
    
    if(orbit3d>0):
        fig2 = go.Figure()
        for p in range(len(outps)):
            steps = len(outps[p][0,:,0])
            if max_steps is None:
                max_steps = steps
            for i in range(nstart,nstart+nparticles):
                for j in range(max_steps):
                    if (outps[p][i,j,1]>psiout):
                        break
                x = outps[p][i,:j,10]
                y = outps[p][i,:j,11]
                z = outps[p][i,:j,12]

                fig2.add_trace(go.Scatter3d(x=x, y=y, z=z,
                               mode='lines',
                               line=dict(width=5),
                               name = labels[p] ))

        fig2.update_layout(title='3D particle orbits',
                      legend = dict(x = 0,y = 0),
                      scene=dict(
                      xaxis_title='X Axis',
                      yaxis_title='Y Axis',
                      zaxis_title='Z Axis'
                      ),
                      width=800, 
                      height=600,
                      margin=dict(l=0, r=0, b=0, t=50))
        fig2.show()
        
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

def show_all_c(trackps,labels,particle_num = 0,steps = 1000,color_layers = None,toroidaln=5):
    if color_layers is not None:
        boundaries = np.arange(0, psiw, psiw/color_layers)  # Set color boundaries from -1 to 1 with intervals of 0.2
        cmap = cm.get_cmap('viridis',boundaries)  # Create a colormap with discrete intervals
        norm = BoundaryNorm(boundaries, cmap.N)
    else:
        cmap = cm.get_cmap('viridis')
        
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize = (30,5))
    for label,trackp in zip(labels,trackps):
        ax1.plot(trackp[particle_num,:steps,0],trackp[particle_num,:steps,1],label = label)
        ax2.scatter(trackp[particle_num,:steps,3],trackp[particle_num,:steps,2]%(2*np.pi),label = label,s=4)
        ax3.scatter(trackp[particle_num,:steps,3],trackp[particle_num,:steps,1],label = label,s=4)
        ax4.scatter(trackp[particle_num,:steps,0],trackp[particle_num,:steps,2],label = label,s=4)
        ax5.plot(trackp[particle_num,:steps,0],trackp[particle_num,:steps,9],label = label)
    ax1.set(title ='$\lambda= $'+str(trackp[particle_num,0,6]/trackp[particle_num,0,5]),xlabel = 't',ylabel = '$\psi$')
    ax2.set(title=particle_num,xlabel = '$\zeta$',ylabel = '$\Theta$')
    ax3.set(xlabel = '$\zeta$',ylabel = '$\psi$')
    ax4.set(xlabel = 't',ylabel = '$\Theta$')
    ax5.set(xlabel = 't',ylabel = 'B')
    plt.legend()
    plt.show()
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize = (30,5))
    for label,trackp in zip(labels,trackps):
        ax1.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,1],trackp[particle_num,:steps,0]),label = label)
        ax2.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,2],trackp[particle_num,:steps,0]),label = label)
        ax3.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,2],trackp[particle_num,:steps,0]),label = label)
        ax4.plot(trackp[particle_num,:steps,0],trackp[particle_num,:steps,13],label = label)
        ax5.plot(trackp[particle_num,:steps,0],trackp[particle_num,:steps,14],label = label)
    ax2.set(xlabel = 't',ylabel = '$d\Theta/dt$',ylim = (-0.005,0.005))
    ax3.set(xlabel = 't',ylabel = '$d\zeta/dt$',ylim = (-0.005,0.005))
    ax4.set(xlabel = 't',ylabel = '$\nablab$')
    ax5.set(xlabel = 't',ylabel = '$\kappa$')
    ax1.set(xlabel = 't',ylabel = '$d\psi/dt$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plots = len(trackps)
    fig, axes = plt.subplots(1, plots, figsize=(7 * plots, 5))
    for ax, trackp, label in zip(axes, trackps, labels):
        if color_layers is not None:
            cax = ax.scatter(trackp[i,:steps,3],trackp[i,:steps,2]%(2*np.pi), c=trackp[i,:steps,1], cmap=cmap,norm=norm, s=5)
        else:
            cax = ax.scatter(trackp[i,:steps,3],trackp[i,:steps,2]%(2*np.pi), c=trackp[i,:steps,1], cmap=cmap, s=5,vmin = 0, vmax = psiw)
        ax.set(xlabel = '$\zeta$',ylabel = '$\Theta$',title = label,xlim =[0,2*np.pi/toroidaln],ylim=[0,2*np.pi])
    fig.colorbar(cax, ax=axes, location='right', shrink=0.8)
    #plt.tight_layout()
    plt.show()
    
def show_all(trackp,label=None,particle_num = 0,steps = 1000,color_layers = 10,psiw = 0.1):
    if color_layers is not None:
        boundaries = np.arange(0, psiw, psiw/color_layers)  # Set color boundaries from -1 to 1 with intervals of 0.2
        cmap = cm.get_cmap('viridis', len(boundaries))  # Create a colormap with discrete intervals
        norm = BoundaryNorm(boundaries, cmap.N)
    else:
        cmap = cm.get_cmap('viridis')
    
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize = (24,5))

    ax1.plot(trackp[particle_num,:steps,0],trackp[particle_num,:steps,1])
    ax2.scatter(trackp[particle_num,:steps,3],trackp[particle_num,:steps,2]%(2*np.pi),s=4)
    ax3.scatter(trackp[particle_num,:steps,3],trackp[particle_num,:steps,1],s=4)
    ax4.scatter(trackp[particle_num,:steps,0],trackp[particle_num,:steps,2],s=4)
    ax1.set(title ='$\lambda= $'+str(trackp[particle_num,0,6]/trackp[particle_num,0,5]),xlabel = 't',ylabel = '$\psi$')
    ax2.set(title=particle_num,xlabel = '$\zeta$',ylabel = '$\Theta$')
    ax3.set(xlabel = '$\zeta$',ylabel = '$\psi$')
    ax4.set(xlabel = 't',ylabel = '$\Theta$')
    
    plt.show()
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (15,5))

    ax1.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,1],trackp[particle_num,:steps,0]))
    ax2.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,2],trackp[particle_num,:steps,0]))
    ax3.plot(trackp[particle_num,:steps,0],np.gradient(trackp[particle_num,:steps,2],trackp[particle_num,:steps,0]))
    ax2.set(xlabel = 't',ylabel = '$d\Theta/dt$',ylim = (-0.005,0.005))
    ax3.set(xlabel = 't',ylabel = '$d\zeta/dt$',ylim = (-0.005,0.005))
    ax1.set(xlabel = 't',ylabel = '$d\psi/dt$')
    
    plt.tight_layout()
    plt.show()
    plots = 1
    fig, axes = plt.subplots(1, plots, figsize=(7 * plots, 5))
    if color_layers is not None:
        cax = axes.scatter(trackp[i,:steps,3],trackp[i,:steps,2]%(2*np.pi), c=trackp[i,:steps,1], cmap=cmap,norm=norm, s=5)
    else:
        cax = axes.scatter(trackp[i,:steps,3],trackp[i,:steps,2]%(2*np.pi), c=trackp[i,:steps,1], cmap=cmap, s=5)
    axes.set_xlabel('$\zeta$')
    axes.set_ylabel('$\Theta$')
    axes.set_title(label)
    cbar = fig.colorbar(cax, ax=axes, location='right', shrink=0.8)
    cbar.set_label('$\psi$')
    #plt.tight_layout()
    plt.show()