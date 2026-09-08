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
import gtc

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
                # diagnosis.F90 writes thermal ion, thermal electron, then
                # fast ion. The previous reader interchanged the latter two.
                data1deseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1])
                data1dfseries[j+i*sect1]=float(lines[7+j+i*(sect1*nspecies+sect2*2)+sect1*2])
        
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
  except FileNotFoundError as error:
    case_directory = fname.rsplit('/', 1)[0] if '/' in fname else '.'
    raise FileNotFoundError(
      f"data1d.read could not find '{fname}'. "
      f"Check that the case directory exists and is spelled correctly: "
      f"'{case_directory}'"
    ) from error
  except:
    print("Something went wrong!")
    return


def neoclassical_current_components(case_path, nt=0, diagnostic=3,
                                     qion=1.0, qfast=1.0, qelectron=-1.0):
    """Return species fluxes and their charge-weighted radial-current sum.

    diagnostic=3 is the smoothed radial particle flux used by the Er solver.
    For files produced with the extended diagnostics, indices 5:12 contain
    the signed and raw fast-ion flux components.
    """
    data1di, data1df, data1de, _, _ = read(
        case_path.rstrip('/') + '/data1d.out', nt=nt)
    ion_flux = data1di[:, :, diagnostic]
    fast_flux = data1df[:, :, diagnostic]
    electron_flux = data1de[:, :, diagnostic]
    return {
        'thermal_ion': ion_flux,
        'fast_ion': fast_flux,
        'electron': electron_flux,
        'total_current': (
            qion * ion_flux + qfast * fast_flux + qelectron * electron_flux
        ),
    }


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

def compare_profile_vs_time(case_paths, ndstep,ndstart = 0, diagnostic=4, species=0,
                     psi_trim_low = 0, psi_trim_high = 5, title=None, figsize_per_row=4, dpi=120,
                     shared_colorbar=True,fontsize=20, tstep = 0.00025,levels = 40, psi=0.5):
    """Compare radial profile evolution across multiple runs.

    case_paths : list of str  — directories containing data1d.out / gtc.out0
    ndstep     : int          — number of timesteps to read
    diagnostic : int          — index into mpdata1d dimension (0=number, 1=energy, 2=momentum, …)
    species    : int          — 0=ion, 1=EP, 2=electron
    psi_trim_low, psi_trim_high : int — number of radial points to trim
    psi        : float        — normalized poloidal-flux location (psi/psi_ped)
                                at which f(psi,t) is plotted. Each run is
                                linearly interpolated from its own radial grid.
    title      : list of str or None  — subplot titles (one per case_path)
    shared_colorbar : bool    — if True, all subplots share the same vmin/vmax
    """
    kind_names = ['ion', 'EP', 'electron']
    ptype_names = [
        'number flux', 'energy flux', 'angular momentum flux',
        'smoothed radial flux', 'Er',
        'smoothed rkdot flux (upara >= 0)',
        'smoothed rkdot flux (upara < 0)',
        'smoothed analytic-vdr flux (upara >= 0)',
        'smoothed analytic-vdr flux (upara < 0)',
        'raw rkdot flux (upara >= 0)',
        'raw rkdot flux (upara < 0)',
        'raw analytic-vdr flux (upara >= 0)',
        'raw analytic-vdr flux (upara < 0)',
    ]

    kind_label = kind_names[species] if species < len(kind_names) else f'kind {species}'
    ptype_label = ptype_names[diagnostic] if diagnostic < len(ptype_names) else f'data index {diagnostic}'

    psi = float(psi)
    if not np.isfinite(psi):
        raise ValueError("psi must be finite")

    n = len(case_paths)
    if title is None:
        title = [f'{kind_label} {ptype_label}'] * n

    # First pass: read data and find global vmin/vmax
    datasets = []
    radial_grids = []
    psi_grids = []
    global_vmin = float('inf')
    global_vmax = float('-inf')
    for case_path in case_paths:
        (data1di, data1df, data1de, field00, fieldrms) = read(
            case_path + "/data1d.out", nt=ndstep)
        species_data = (data1di, data1df, data1de)
        if species < 0 or species >= len(species_data):
            raise ValueError("species must be 0 (ion), 1 (EP), or 2 (electron)")
        # Er is a global field stored in the thermal-ion block. All other
        # diagnostics select the requested species.
        selected_data = data1di if diagnostic == 4 else species_data[species]
        plot_data = selected_data[
            ndstart:, psi_trim_low+1:-(2+psi_trim_high), diagnostic]
        (physical_parameters,radial_grid,radial_profile)=gtc.read(case_path+"/gtc.out0")
        radial_slice = slice(psi_trim_low, -(1+psi_trim_high))
        radial_coordinates = radial_grid[radial_slice, 1]
        psi_coordinates = radial_grid[radial_slice, 2]
        if plot_data.shape[1] != radial_coordinates.size:
            raise ValueError(
                f"Radial-grid mismatch for '{case_path}': data1d has "
                f"{plot_data.shape[1]} retained points but gtc.out0 has "
                f"{radial_coordinates.size}"
            )
        if psi_coordinates.size < 2 or not np.all(np.isfinite(psi_coordinates)):
            raise ValueError(
                f"Need at least two finite radial-grid points for '{case_path}'"
            )
        if not np.all(np.diff(psi_coordinates) > 0.0):
            raise ValueError(
                f"psi/psi_ped grid must be strictly increasing for '{case_path}'"
            )
        if psi < psi_coordinates[0] or psi > psi_coordinates[-1]:
            raise ValueError(
                f"Requested psi/psi_ped={psi:g} is outside the retained "
                f"range [{psi_coordinates[0]:g}, {psi_coordinates[-1]:g}] "
                f"for '{case_path}'"
            )
        radial_grids.append(radial_coordinates)
        psi_grids.append(psi_coordinates)
        datasets.append(plot_data)
        if shared_colorbar:
            global_vmin = min(global_vmin, plot_data.min())
            global_vmax = max(global_vmax, plot_data.max())

    # Second pass: plot with consistent color scale
    fig, axes = plt.subplots(1, n, figsize=(8 * n + 3, 6), dpi=dpi, squeeze=False)

    contour_kw = {}
    if shared_colorbar:
        shared_levels = np.linspace(global_vmin, global_vmax, levels+1)
        contour_kw = {'levels': shared_levels}
    else:
        contour_kw = {'levels': levels}

    for i, plot_data in enumerate(datasets):
        ax = axes[0, i]
        cf = ax.contourf(
            radial_grids[i], np.arange(plot_data.shape[0])*tstep, plot_data,
            cmap='jet', **contour_kw)
        evaluation_radius = np.interp(psi, psi_grids[i], radial_grids[i])
        ax.axvline(
            evaluation_radius, color='white', linestyle='--', linewidth=2.0
        )
        ax.set(xlabel='$r/a$', ylabel='t $R_0/C_s$', title=title[i])
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.title.set_fontsize(fontsize)
        fig.colorbar(cf, ax=ax)
        

    fig.tight_layout()
    plt.show()

    profile_fig, profile_ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for i, plot_data in enumerate(datasets):
        profile_ax.plot(radial_grids[i], plot_data[-1, :], label=title[i])
    profile_ax.legend(fontsize=fontsize)
    profile_ax.set_xlabel('r/a', fontsize=fontsize)
    ylabel = 'Er kV/m' if diagnostic == 4 else ptype_label
    profile_ax.set_ylabel(ylabel, fontsize=fontsize)
    profile_fig.tight_layout()
    plt.title('Radial profile at last timestep', fontsize=fontsize)
    plt.show()

    profile_fig, profile_ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    first_step_for_average = int((ndstep-ndstart)/2)
    for i, plot_data in enumerate(datasets):
        profile_ax.plot(radial_grids[i], np.average(plot_data[first_step_for_average:, :], axis=0), label=title[i])
    profile_ax.legend(fontsize=fontsize)
    profile_ax.set_xlabel('r/a', fontsize=fontsize)
    ylabel = 'Er kV/m' if diagnostic == 4 else ptype_label
    profile_ax.set_ylabel(ylabel, fontsize=fontsize)
    profile_fig.tight_layout()
    plt.title('Average radial profile', fontsize=fontsize)
    plt.show()

    time_fig, time_ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    for i, plot_data in enumerate(datasets):
        time_trace = np.array([
            np.interp(psi, psi_grids[i], radial_profile)
            for radial_profile in plot_data
        ])
        time_ax.plot(
            np.arange(plot_data.shape[0])*tstep, time_trace, label=title[i]
        )
    time_ax.legend(fontsize=fontsize)
    time_ax.set_xlabel('t $R_0/C_s$', fontsize=fontsize)
    time_ax.set_ylabel(ylabel, fontsize=fontsize)
    time_ax.set_title(
        rf'time plot at $\psi/\psi_{{ped}}={psi:g}$', fontsize=fontsize
    )
    time_fig.tight_layout()
    plt.show()

    return fig,plt
