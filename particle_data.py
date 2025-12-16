import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.dataframe as dd
import equilibrium as Eq
import stellarator as stl
import spline as sp
from matplotlib.gridspec import GridSpec

def read_full_parts_dask(path,mpis,old11=False,directory='full_parts/'):
    ''''reads in the particle data from the full_parts directory'''
    dfs = []

    for i in range(mpis):
        filename = f"parts{str(i).zfill(5)}.out"
        df = dd.read_csv(f"{path}{directory}{filename}", header=None, assume_missing=True)
        
        dfs.append(df)

    dfs = dd.concat(dfs, ignore_index=True)
    
    # Convert Dask DataFrame back to Pandas DataFrame for operations not supported by Dask
    df = dfs.compute(errors= 'ignore')
    if old11:
        df =clean_dask_old11(df)
    else:
        df = clean_dask(df)
    print("df created from data at: ",path,directory)
    return df



def clean_dask(dfs):
    dfs = dfs.rename(columns={0:'psi_f',1:'theta_f', 2:'zeta_f', 3:'lambda_f', 4:'marker_weight', 5:'E_f', 6:'particle_weight', 7:'psi_i', 8:'theta_i', 9:'zeta_i', 10:'lambda_i', 11:'E_i'})
    dfs = dfs.drop(columns = [12])
    dfs['dpsi'] = dfs['psi_f'] - dfs['psi_i']
    dfs['dpsi2']=dfs['dpsi']**2
    
    #dfs = dfs[dfs['lambda_f'] >= -1]
    lost = dfs[dfs['marker_weight'] == 0]
    kept = dfs[dfs['marker_weight'] == 1]

    return dfs, lost, kept

def clean_dask_old11(dfs):
    dfs = dfs.rename(columns={0:'psi_f',1:'theta_f', 2:'zeta_f', 3:'lambda_f', 4:'marker_weight', 5:'E_f', 6:'psi_i', 7:'theta_i', 8:'zeta_i', 9:'lambda_i', 10:'E_i'})
    dfs = dfs.drop(columns = [11])
    dfs['dpsi'] = dfs['psi_f'] - dfs['psi_i']
    dfs['dpsi2']=dfs['dpsi']**2
    
    #dfs = dfs[dfs['lambda_f'] >= -1]
    lost = dfs[dfs['marker_weight'] == 0]
    kept = dfs[dfs['marker_weight'] == 1]

    return dfs, lost, kept

def plot2d(x,xdiv,y,ydiv,d1, scalefactor = 1,ratio=1,kept = 0,nbins=200,ax = None,psiw=1,labels = 'natural',title = None,figsize= None,lables = True,clabels = True,old11=False):
    df1 = d1[kept+1]
    if (ratio !=3):
        df_div1 = d1[0]
    elif (ratio == 3):
        df_div1 = d1[kept+1]
        
    if xdiv == 's':
        counts_div1, xedges_1, yedges_1 = np.histogram2d(np.sqrt(df_div1[x]/psiw)*np.cos(df_div1[y]),np.sqrt(df_div1[x]/psiw)*np.sin(df_div1[y]),bins=nbins)   
        counts_1, _,_ = np.histogram2d(np.sqrt(df1[x]/psiw)*np.cos(df1[y]),np.sqrt(df1[x]/psiw)*np.sin(df1[y]), bins=[xedges_1, yedges_1]) # Use the same bins
    else:
        if old11:
            if ratio == 0:
                counts_1,xedges_1, yedges_1 = np.histogram2d(df1[x], df1[y], bins=nbins) # Use the same bins
            elif ratio ==2:
                counts_div1, xedges_1, yedges_1 = np.histogram2d(df_div1[xdiv],df_div1[ydiv],bins=nbins)
            else:
                counts_div1, xedges_1, yedges_1 = np.histogram2d(df_div1[xdiv],df_div1[ydiv],bins=nbins)
                counts_1, _,_ = np.histogram2d(df1[x], df1[y], bins=[xedges_1, yedges_1]) # Use the same bins
        else:
            if ratio == 0:
                counts_1,xedges_1, yedges_1 = np.histogram2d(df1[x], df1[y], bins=nbins,weights =df1['particle_weight']) # Use the same bins
            elif ratio ==2:
                counts_div1, xedges_1, yedges_1 = np.histogram2d(df_div1[xdiv],df_div1[ydiv],bins=nbins,weights =df_div1['particle_weight'])
            else:
                counts_div1, xedges_1, yedges_1 = np.histogram2d(df_div1[xdiv],df_div1[ydiv],bins=nbins,weights =df_div1['particle_weight'])
                counts_1, _,_ = np.histogram2d(df1[x], df1[y], bins=[xedges_1, yedges_1],weights =df1['particle_weight']) # Use the same bins

    if x=='theta_i' or x=='zeta_i' or x == 'theta_f' or x=='zeta_f':
        xedges_1 = xedges_1/(2*np.pi)
        
    if y=='theta_i' or y=='zeta_i' or y == 'theta_f' or y=='zeta_f':
        yedges_1 = yedges_1/(2*np.pi)
        
    # Calculate the ratio f[x,y]/g[x,y]
    if (ratio ==1):
        ratio1 = counts_1/ (counts_div1 + 1e-8)
        clabel ='$\\Delta n/n$'
    elif (ratio ==2):
        ratio1 = counts_div1
        clabel = 'n'
    elif (ratio ==3):
        ratio1 = counts_1- counts_div1
        clabel = '$\\Delta n$ of confined particles'
    else:
        ratio1 = counts_1*scalefactor
        clabel = ''
    
    if ax is None:
        ax = plt.gca()
    
    if figsize is not None:    
        ax.figure(figsize=(figsize[0], figsize[1]))
    if ratio == 1:
        img = ax.imshow(ratio1.T, origin='lower', aspect='auto', 
                    extent=[xedges_1[0], xedges_1[-1], yedges_1[0], yedges_1[-1]], 
                    cmap='jet',vmin = 0,vmax = 1)
    else:
        img = ax.imshow(ratio1.T, origin='lower', aspect='auto', 
                    extent=[xedges_1[0], xedges_1[-1], yedges_1[0], yedges_1[-1]], 
                    cmap='jet')
    fig = ax.get_figure()
    if labels:
        if (y == 'E_i' or y=='E_f'):
            yticks = ax.get_yticks()
            ax.set_yticklabels([f'{tick/1000:.1f}' for tick in yticks])
        if (x == 'E_i' or x=='E_f'):
            xticks = ax.get_xticks()
            ax.set_xticklabels([f'{tick/1000:.1f}' for tick in xticks])

        if (labels=='natural'):
            ax.set_xlabel(x)  # Replace 'X' with the appropriate label
            ax.set_ylabel(y)  # Replace 'Y' with the appropriate label
        elif (xdiv=='s'):
            ax.set_xlabel('$\\sqrt{\\psi_p}cos(\\Theta)$')  # Replace 'X' with the appropriate label
            ax.set_ylabel('$\\sqrt{\\psi_p}sin(\\Theta)$')
        elif labels is not None:
            ax.xlabel(labels[0],fontsize = 20)
            ax.ylabel(labels[1],fontsize = 20)
        if title is not None:
            ax.title(title,fontsize = 20)
        # Adjust layout to prevent overlap
    else:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    if clabels:
        fig.colorbar(img, ax=ax, label=clabel)
    fig.tight_layout()
    return img

def plotfluxsurface(x, xdiv, y, ydiv, d1, psi=None, dpsi=None, theta=None, dtheta=None, zeta=None, dzeta=None,E=None, dE=None,lamb=None, dlamb=None,
                    scalefactor=1, ratio=1, kept=0, nbins=200, title=None, xlabel=None, ylabel=None,
                    qsp=None, b_df=None, bsp=None, ax=None,psiw=1,labels = True,clabels = True,toroidaln = 5,old11=False):
    if ax is None:
        ax = plt.gca()  # Use the current axis if none is provided
    if dpsi is not None:
        dpsi = dpsi*psiw
    df1 = [None] * 3
    for i in range(len(d1)):
        df = d1[i]
        if dpsi is not None:
            df = df[(df['psi_i'] < psi + dpsi / 2) & (df['psi_i'] > psi - dpsi / 2)]
        if dtheta is not None:
            df = df[(df['theta_i'] < theta + dtheta / 2) & (df['theta_i'] > theta - dtheta / 2)]
        if dzeta is not None:
            df = df[(df['zeta_i'] < zeta + dzeta / 2) & (df['zeta_i'] > zeta - dzeta / 2)]
        if dE is not None:
            df = df[(df['E_i'] < E + dE / 2) & (df['E_i'] > E - dE / 2)]
        if dlamb is not None:
            df = df[(df['lambda_i'] < lamb + dlamb / 2) & (df['lambda_i'] > lamb - dlamb / 2)]
        df1[i] = df

    # Plot within the given axis
    
    p2d = plot2d(x, xdiv, y, ydiv, df1, scalefactor=scalefactor, ratio=ratio, psiw = psiw,kept=kept, nbins=nbins, ax=ax,labels = labels,clabels = clabels,old11=old11)


    if qsp is not None:
        q = qsp(psi)
        zmax = max(d1[0]['zeta_i'])
        x = np.linspace(0, zmax, 100)
        ax.plot(x/(2*np.pi), (x / q)/(2*np.pi), color='green',lw =3)
        ax.plot(x/(2*np.pi), (np.pi / 2 + x / q)/(2*np.pi), color='green',lw = 3)
        ax.plot(x/(2*np.pi), (np.pi + x / q)/(2*np.pi), color='green',lw =3)
        ax.plot(x/(2*np.pi), (3 * np.pi / 2 + x / q)/(2*np.pi), color='green',lw =3)
        ax.set_ylim(0, 1)
    if b_df is not None:
        b_plot=b_contour(b_df, ax=ax)
    if bsp is not None:
        b_plot=sp.contour_sp(bsp, toroidaln,psi = psi,zeta = zeta,psiw=psiw, ax=ax,labels = clabels,contour_levels = 10,linewidth = 2)
    if labels:
        ax.set_title(f'$\\psi_n$ = {psi} $\\Theta$ = {theta} $\\zeta$ = {zeta}')
        if title is not None:
            ax.set_title(title,fontsize = 24)
        if xlabel is not None:
            ax.set_xlabel(xlabel,fontsize = 15)
        if ylabel is not None:
            ax.set_ylabel(ylabel,fontsize = 15)

    
        if x == 'lambda_i':
            ax.set_xlim(0, 1.5)
            ax.set_xlabel('$\\lambda = \\mu/E$',fontsize = 15)

    if bsp is not None:
        return p2d,b_plot
    else:
        return p2d,None

def plotfluxsurface_m(x, xdiv, y, ydiv, dfs, psi=None, dpsi=None, theta=None, dtheta=None, zeta=None, dzeta=None,
                      E=None, dE=None, lamb=None, dlamb=None, scalefactor=1, ratio=1, kept=0, nbins=200, titles=None,
                      xlabel=None, ylabel=None, qsp=None, b_df=None, bsp=None, psiw=1,fontsize = 20,x_lim = None,old11=False):
    plots = len(dfs)
    if isinstance(psi, (list, tuple, np.ndarray)):
        rows = len(psi)
    else:
        rows = 1

    # Create a GridSpec layout with extra space for colorbars
    fig = plt.figure(figsize=(8 * plots + 3, 6 * rows))
    if bsp is not None:
        spec = GridSpec(rows, 2 * plots + 2, figure=fig, width_ratios=[1] * (2 * plots) + [0.15,0.45], height_ratios=[1] * rows)
    else:
        spec = GridSpec(rows, 2 * plots, figure=fig, width_ratios=[1] * (2 * plots) , height_ratios=[1] * rows)
    
    axes = []
    for row_idx in range(rows):
        row_axes = []
        for col_idx in range(plots):
            ax = fig.add_subplot(spec[row_idx, 2 * col_idx:2 * col_idx + 2])
            row_axes.append(ax)
        axes.append(row_axes)

    # Store mappable objects for the colorbars
    mappable_plot2d = None
    mappable_b_contour = None

    if (ratio ==1):
        clabel ='$\\Delta n/n$'
    elif (ratio ==2):
        clabel = 'n'
    elif (ratio ==3):
        clabel = '$\\Delta n$ of confined particles'
    else:
        clabel = ''
    
    for row_idx, row_axes in enumerate(axes):
        if psi is not None: 
            current_psi = psi[row_idx]*psiw if rows > 1 else psi*psiw
        else:
            current_psi = None
        for col_idx, ax in enumerate(row_axes):
            df = dfs[col_idx]
            title = titles[col_idx] if titles else None
            mappable_plot2d,mappable_b_contour = plotfluxsurface(
                x, xdiv, y, ydiv, df, current_psi, dpsi, theta, dtheta, zeta, dzeta, E, dE, lamb, dlamb,
                scalefactor, ratio, kept, nbins, title, xlabel, ylabel,
                qsp, b_df, bsp = bsp, ax=ax, psiw=psiw, labels=False, clabels=False,old11=old11
            )
            
            if x_lim is not None:
                ax.set_xlim(x_lim[0],x_lim[1])
            # Add y-axis labels and ticks only to the leftmost column
            if col_idx == 0:
                if isinstance(psi, (list, tuple, np.ndarray)):
                    ax.set_ylabel(' $\\psi$ = '+str(psi[row_idx])+'\n\n'+ylabel, fontsize=fontsize)
                else:
                    ax.set_ylabel(ylabel,fontsize=fontsize)
                ax.tick_params(axis='y', which='both')
                yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5)
                ax.set_yticks(yticks)
                if (y=='E_i' or y == 'E_f'):
                    ax.set_yticklabels([f'{tick/1000:.0f}' for tick in yticks],fontsize=fontsize)
                elif (y=='psi_i' or y == 'psi_f'):
                    ax.set_yticklabels([f'{tick/psiw:.1f}' for tick in yticks],fontsize=fontsize)
                else:
                    ax.set_yticklabels([f'{tick:.3f}' for tick in yticks],fontsize=fontsize)
            else:
                ax.set_yticklabels([])
                
            if row_idx ==0:
                ax.set_title(title,fontsize = fontsize)
            # Add x-axis labels and ticks only to the bottom row
            if row_idx == rows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                ax.tick_params(axis='x', which='both')
                xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=5) 
                if col_idx!=0:
                    xticks = xticks[1:]
                ax.set_xticks(xticks)
                if (x=='E_i' or x == 'E_f'):
                    ax.set_xticklabels([f'{tick/1000:.0f}' for tick in yticks],fontsize=fontsize)
                elif (x=='psi_i' or x == 'psi_f'):
                    ax.set_xticklabels([f'{tick/psiw:.0f}' for tick in yticks],fontsize=fontsize)
                else:
                    ax.set_xticklabels([f'{tick:.2f}' for tick in xticks],fontsize = fontsize)
            else:
                ax.set_xticklabels([])

        if bsp is not None:
            #cbar_ax2 = fig.add_subplot(spec[row_idx, -4])
            cbar_ax1 = fig.add_subplot(spec[row_idx, -2])  # Colorbar for b_contour

            cbar = fig.colorbar(mappable_b_contour, cax=cbar_ax1, orientation='vertical', format='%.1f', spacing='proportional')
            cbar.set_label(label='$B/B_0$', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

            # Set fewer ticks
            num_ticks = 5  # Adjust for more/less ticks
            ticks = np.linspace(cbar.vmin, cbar.vmax, num_ticks)
            cbar.set_ticks(ticks)

        if mappable_plot2d:
            cbar_ax = fig.add_axes([1.02, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
            cbar = fig.colorbar(mappable_plot2d, cax=cbar_ax, orientation='vertical', format='%.1f')
            cbar.set_label(label=clabel, fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)
            
            
            
            #cbar = fig.colorbar(mappable_plot2d, cax=cbar_ax2, orientation='vertical', format='%.1f')
            #cbar.set_label(label=clabel, fontsize=fontsize)
            #cbar.ax.tick_params(labelsize=fontsize)

            # Set fewer ticks
            num_ticks = 5
            ticks = np.linspace(cbar.vmin, cbar.vmax, num_ticks)
            cbar.set_ticks(ticks)
    # Adjust spacing between plots and colorbars
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def plot_ratio1d(x,xdiv,binnum,d1,label='',scalefactor = 1,kept=0,ratio =1,psiw=1,lw=1):
    df1 = d1[kept+1]
    df_div1 = d1[0]
    
    counts_df1, bin_edges_df1 = np.histogram(df1[x], bins=binnum)
    counts_df_div1, bin_edges_df_div = np.histogram(df_div1[xdiv], bins=bin_edges_df1)
    
    if ratio ==1:
        ratio_1 = counts_df1 / (counts_df_div1 + 1e-8)
    elif (ratio ==2):
        ratio_1 = counts_df_div1*scalefactor
    else:
        ratio_1 = counts_df1 *scalefactor
        
    # Calculate bin centers from bin edges
    bin_centers = (bin_edges_df1[:-1] + bin_edges_df1[1:]) / 2
    
    # Plot the ratio
    if x == 'psi_i' or x =='psi_f':
        plt.plot(bin_centers/psiw, ratio_1, linestyle='-',label = label,lw=lw)
    else:
        plt.plot(bin_centers, ratio_1, linestyle='-',label = label,lw=lw)
    plt.legend()
