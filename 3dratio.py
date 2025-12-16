
def plot3d_ratio_cart(d1, bins=50,gs = 60j, scalefactor=1, ratio=1):
    """
    Plot the 3D ratio of lost particles to all particles across (psi, theta, zeta) space.
    
    Parameters:
    - d1: List containing three DataFrames [all_particles, lost_particles, confined_particles]
    - bins: Number of bins in each dimension for the histogram
    - scalefactor: Scaling factor for data normalization if needed
    - ratio: Determines the type of plot (1=ratio, 2=counts, 3=confined-loss difference)
    """
    # Unpack DataFrames
    df_all = d1[0]
    df_lost = d1[1]
    
    # Define columns (assume DataFrame has columns named 'psi', 'theta', 'zeta')
    psi_all, theta_all, zeta_all = df_all['psi_i'], df_all['theta_i'], df_all['zeta_i']
    psi_lost, theta_lost, zeta_lost = df_lost['psi_i'], df_lost['theta_i'], df_lost['zeta_i']
    
    # 3D histogram counts
    counts_all, edges = np.histogramdd((psi_all, theta_all, zeta_all), bins=bins)
    counts_lost, _ = np.histogramdd((psi_lost, theta_lost, zeta_lost), bins=edges)
    
    # Calculate the ratio based on the chosen mode
    if ratio == 1:
        ratio_counts = counts_lost / (counts_all + 1e-8)
        clabel = '$\\Delta n/n$'
    elif ratio == 2:
        ratio_counts = counts_lost
        clabel = 'n lost'
    elif ratio == 3:
        ratio_counts = counts_all - counts_lost
        clabel = '$\\Delta n$ of confined particles'
    else:
        ratio_counts = counts_all * scalefactor
        clabel = ''
    
    # Define the centers of bins for plotting
    psi_centers = (edges[0][1:] + edges[0][:-1]) / 2
    theta_centers = (edges[1][1:] + edges[1][:-1]) / 2
    zeta_centers = (edges[2][1:] + edges[2][:-1]) / 2
    
    # Flatten the 3D data for plotting
    psi_flat, theta_flat, zeta_flat = np.meshgrid(psi_centers, theta_centers, zeta_centers, indexing='ij')
    psi_flat = psi_flat.flatten()
    theta_flat = theta_flat.flatten()
    zeta_flat = zeta_flat.flatten()
    ratio_flat = ratio_counts.flatten()

    # Prepare arrays for Cartesian coordinates and magnetic field values
    x = np.zeros(len(psi_flat))
    y = np.zeros(len(psi_flat))
    z = np.zeros(len(psi_flat))

    
    # Convert (psi, theta, zeta) to Cartesian coordinates
    for i in range(len(psi_flat)):
        x[i], y[i], z[i] = cartesian_boozer(psi_flat[i], theta_flat[i], zeta_flat[i])
    
    # Data coordinates
    points = np.array( (z.flatten(), x.flatten(), y.flatten()) ).T
    # Data values @ above coords
    values = ratio_flat

    # Grid to interp to
    X, Y,Z = np.mgrid[ x.min():x.max():gs, y.min():y.max():gs,z.min():z.max():gs]
    # New data values on interp grid
    newdata = griddata( points, values, (Z,X,Y), method = 'linear')

    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    newdata = newdata.flatten()
    # Generate the mask
    mask = generate_mask_parallel(X, Y, Z, psiw)

    # Apply the mask
    masked_newdata = apply_mask(newdata, mask)
    
    '''
    for i in range(len(newdata)):
        pdum = inverse_cartesian(X[i],Y[i],Z[i])[0]
        if pdum>=psiw:
            newdata[i]=np.nan
       
    # Parallelized function
    def check_and_mask(index):
        pdum = inverse_cartesian(X[index], Y[index], Z[index])
        if pdum[0] >= psiw*0.2 and pdum[0] <=psiw*0.8:
            return np.nan
        else:
            return newdata[index]

    # Run in parallel
    newdata = np.array(Parallel(n_jobs=-1)(delayed(check_and_mask)(i) for i in range(len(newdata))))
    '''
    # Plot the isosurface of particle ratio
    fig = go.Figure(data=[go.Volume(
        x=X.flatten(),y=Y.flatten(),z=Z.flatten(),
        value=masked_newdata,
        opacity=0.7,  # Adjust for transparency
        isomin = 0.1,
        isomax = 0.5,
        surface_count=3,  # Controls number of isosurfaces
        colorscale='amp',  # Color scale for density representation
        colorbar=dict(title=clabel),
    )])
    # Customize layout for better visualization
    fig.update_layout(scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectratio=dict(x=1, y=1, z=1)
    ))
    return fig

# Define check_condition at the top level
def check_condition(index, X, Y, Z, psiw):
    pdum = inverse_cartesian(X[index], Y[index], Z[index])
    return pdum[0] <= psiw * 0.2 or pdum[0] >=psiw*0.8

# Modify generate_mask_parallel to use this function
def generate_mask_parallel(X, Y, Z, psiw):
    with Pool() as pool:
        results = pool.starmap(
            check_condition,
            [(i, X, Y, Z, psiw) for i in range(len(X))]
        )
    return np.array(results, dtype=bool)

def apply_mask(data, mask):
    # Apply the mask to filter or process the dataset
    return np.where(mask, np.nan, data)