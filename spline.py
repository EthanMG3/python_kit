#=================================
#====== Spline mod ==========
#-----------content---------------
# >> makes all the spline functions
# >> reads some of the spline functions
#---------------------------------
# --> Ethan Green
# --> E-mail: ethanmg@uci.edu
# --> 10/18/2024
#=================================

import numpy as np
import matplotlib.pyplot as plt

def spline0(pdum, nsp, delx, y):
    """
    Python translation of the Fortran function spline0.
    
    Parameters:
    pdum : float, the point at which to evaluate the spline
    nsp : int, the number of points in the spline
    delx : float, the step size
    y : 2D numpy array of shape (3, nsp), where:
        y[0, :] contains the function values,
        y[1, :] contains the first derivatives,
        y[2, :] contains the second derivatives.
    
    Returns:
    spline0 : float, the evaluated spline value at pdum
    """
    
    # Locate the index i
    i = max(1, min(nsp - 1, int(np.ceil(pdum / delx))))

    # Calculate the offset dpx
    dpx = pdum - delx * (i - 1)

    # Evaluate the spline at pdum
    spline0_value = y[0, i] + dpx * y[1, i] + dpx * dpx * y[2, i]

    return spline0_value

def spline1(pdum, nsp, delx, y):
    """
    Python translation of the Fortran function spline1.
    
    Parameters:
    pdum : float, the point at which to evaluate the spline
    nsp : int, the number of points in the spline
    delx : float, the step size
    y : 2D numpy array of shape (3, nsp), where:
        y[0, :] contains the function values,
        y[1, :] contains the first derivatives,
        y[2, :] contains the second derivatives.
    
    Returns:
    spline1_value : float, the evaluated spline value at pdum
    """
    # Locate the index i
    i = int(pdum / delx)
    dpx = pdum - delx * i

    # Apply sqrt(x) expansion near x = 0 if i == 1
    if i == 0:
        dpx = np.sqrt(dpx)
    dp2 = dpx * dpx  # Square of dpx
    # Evaluate the spline at pdum
    spline1_value = y[0, i] + dpx * y[1, i] + dp2 * y[2, i]
    return spline1_value


def spline3d(x, y, z, deriv, nx, ny, nz, delx, dely, delz, f, bcx, bcy, bcz):
    """
    Python translation of the Fortran subroutine spline3d.
    
    Parameters:
    x, y, z : float, the coordinates at which to evaluate the spline
    deriv : int, derivative flag (0 for function value, 1 for dx, 2 for dy, 3 for dz)
    nx, ny, nz : int, the number of grid points along the x, y, z axes
    delx, dely, delz : float, step sizes along x, y, z axes
    f : 4D numpy array of shape (27, nx, ny, nz), the spline data
    bcx, bcy, bcz : int, boundary condition flags for x, y, z (1 for special, 2 for periodic)
    GTC_ABORT : function, to handle error conditions
    """
    
    # Locate the indices for the spline cell (no out-of-range checking)
    i = int(x / delx)
    j = int(y / dely)
    k = int(z / delz)
    
    dx = x - delx * i
    dy = y - dely * j
    dz = z - delz * k
    


    # Apply boundary condition (bcx, bcy, bcz) if needed
    if i == 0 and bcx == 1:
        dx = np.sqrt(dx)
    if j == 0 and bcy == 1:
        dy = np.sqrt(dy)
    if k == 0 and bcz == 1:
        dz = np.sqrt(dz)

    #dxinv = 1.0 / dx
    #dyinv = 1.0 / dy
    #dzinv = 1.0 / dz

    # Initialize the spline vector
    dvec = np.zeros(27)

    # Handle the deriv flag
    if deriv == 0:  # No derivative, evaluate the function
        dvec[0] = 1.0
        dvec[1] = dx
        dvec[2] = dx * dx
        dvec[3:6] = dvec[0:3] * dy
        dvec[6:9] = dvec[3:6] * dy
        dvec[9:18] = dvec[0:9] * dz
        dvec[18:27] = dvec[9:18] * dz

    elif deriv == 1:  # Partial derivative in x
        dvec[2] = 2.0 * dx
        dvec[3:6] = dvec[0:3] * dy
        dvec[6:9] = dvec[3:6] * dy
        dvec[9:18] = dvec[0:9] * dz
        dvec[18:27] = dvec[9:18] * dz

    elif deriv == 2:  # Partial derivative in y
        if bcy == 1 and j == 1:
            dvec[0:3] = 0.0
            dvec[6] = 1.0
            dvec[7] = dx
            dvec[8] = dx * dx
            dvec[3:6] = dvec[6:9] * 0.5 * dyinv
            dvec[9:18] = dvec[0:9] * dz
            dvec[18:27] = dvec[9:18] * dz
        else:
            dvec[0:3] = 0.0
            dvec[3] = 1.0
            dvec[4] = dx
            dvec[5] = dx * dx
            dvec[6:9] = dvec[3:6] * 2.0 * dy
            dvec[9:18] = dvec[0:9] * dz
            dvec[18:27] = dvec[9:18] * dz

    elif deriv == 3:  # Partial derivative in z
        if bcz == 1 and k == 1:
            dvec[0:9] = 0.0
            dvec[18] = 1.0
            dvec[19] = dx
            dvec[20] = dx * dx
            dvec[21:24] = dvec[18:21] * dy
            dvec[24:27] = dvec[21:24] * dy
            dvec[9:18] = dvec[18:27] * dzinv * 0.5
        else:
            dvec[0:9] = 0.0
            dvec[9] = 1.0
            dvec[10] = dx
            dvec[11] = dx * dx
            dvec[12:15] = dvec[9:12] * dy
            dvec[15:18] = dvec[12:15] * dy
            dvec[18:27] = dvec[9:18] * dz * 2.0

    else:
        print(f"Spline3d Error: wrong derivative flag deriv={deriv}")
        GTC_ABORT()

    # Compute the spline value by taking the dot product
    return np.sum(f[:, i, j, k] * dvec)


def construct_spline0(iflag, nsp, delx, y):
    """
    Python translation of the Fortran subroutine construct_spline0.
    
    Parameters:
    iflag : int, specifies the type of interpolation for the first point
    nsp : int, number of points
    delx : float, step size
    y : 2D numpy array of shape (3, nsp), where:
        y[0, :] contains the y-values,
        y[1, :] will contain the first derivatives,
        y[2, :] will contain the second derivatives.
    """

    # First point
    if iflag == 0 or iflag == 3:
        # iflag=0: first point being y = y1 + y2*x + y3*x*x
        y[1, 0] = (4.0 * y[0, 1] - y[0, 2] - 3.0 * y[0, 0]) / (2.0 * delx)
        y[2, 0] = (y[0, 1] - y[0, 0] - y[1, 0] * delx) / (delx * delx)

    elif iflag == 1:
        # iflag=1: first point being linear function y = y1 + y2*x
        y[1, 0] = (y[0, 1] - y[0, 0]) / delx
        y[2, 0] = 0.0

    elif iflag == 2:
        # iflag=2: first point being quadratic function y = y1 + y3*x*x
        y[1, 0] = 0.0
        y[2, 0] = (y[0, 1] - y[0, 0]) / (delx * delx)

    # Process for smoothing or not based on iflag==3
    if iflag == 3:
        # do not smooth f1
        for i in range(1, nsp - 2):
            ipp = min(i + 2, nsp - 1)
            y[1, i] = -y[1, i - 1] + 2.0 * (y[0, i] - y[0, i - 1]) / delx
    else:
        # smooth f1
        for i in range(1, nsp - 2):
            ipp = min(i + 2, nsp - 1)
            y[1, i] = -y[1, i - 1] + 2.0 * (y[0, i] - y[0, i - 1]) / delx

            # smooth the function
            y[0, i + 1] = 0.5 * delx * y[1, i] + 0.25 * y[0, ipp] + 0.75 * y[0, i]

    # Last two points for the first derivatives
    y[1, nsp - 2] = -y[1, nsp - 3] + 2.0 * (y[0, nsp - 2] - y[0, nsp - 3]) / delx
    y[1, nsp - 1] = -y[1, nsp - 2] + 2.0 * (y[0, nsp - 1] - y[0, nsp - 2]) / delx

    # Compute second derivatives for inner points
    for i in range(1, nsp - 1):
        y[2, i] = (y[1, i + 1] - y[1, i]) / (2.0 * delx)

    # Last point second derivative is not used
    y[2, nsp - 1] = 0.0


def construct_spline1(nsp, delx, f):
    """
    Python translation of the Fortran subroutine construct_spline1.
    
    Parameters:
    nsp : int, number of points
    delx : float, step size
    f : 2D numpy array of shape (3, nsp), where:
        f[0, :] contains the function values,
        f[1, :] will contain the first derivatives,
        f[2, :] will contain the second derivatives.
    """
    
    # First point
    f[1, 0] = (2.0 * f[0, 1] - f[0, 2] - f[0, 0]) / ((2.0 - np.sqrt(2.0)) * np.sqrt(delx))
    f[2, 0] = (f[0, 1] - f[0, 0] - f[1, 0] * np.sqrt(delx)) / delx

    # Second point
    f[1, 1] = 0.5 * f[1, 0] / np.sqrt(delx) + f[2, 0]

    # Smooth third point
    f[0, 2] = 0.5 * delx * f[1, 1] + 0.25 * f[0, 3] + 0.75 * f[0, 1]
    f[2, 1] = (f[0, 2] - f[0, 1] - delx * f[1, 1]) / (delx * delx)

    # Iterate over the remaining points
    for i in range(2, nsp - 2):
        ipp = min(i + 2, nsp - 1)
        f[1, i] = -f[1, i - 1] + 2.0 * (f[0, i] - f[0, i - 1]) / delx

        # Smooth the next point
        f[0, i + 1] = 0.5 * delx * f[1, i] + 0.25 * f[0, ipp] + 0.75 * f[0, i]

    # Final points
    f[1, nsp - 2] = -f[1, nsp - 3] + 2.0 * (f[0, nsp - 2] - f[0, nsp - 3]) / delx
    f[1, nsp - 1] = -f[1, nsp - 2] + 2.0 * (f[0, nsp - 1] - f[0, nsp - 2]) / delx

    # Calculate second derivatives for the inner points
    for i in range(2, nsp - 1):  # Start from i=2 (index 1 in Python)
        f[2, i] = (f[1, i + 1] - f[1, i]) / (2.0 * delx)

    # Last point second derivative is not used
    f[2, nsp - 1] = 0.0

def construct_spline_periodic(nsp, delx, y):
    """
    Python translation of the Fortran subroutine construct_spline_periodic.
    
    Parameters:
    nsp : int, number of points (must be even for periodic spline)
    delx : float, step size
    y : 2D numpy array of shape (3, nsp), where:
        y[0, :] contains the function values,
        y[1, :] will contain the first derivatives,
        y[2, :] will contain the second derivatives.
    GTC_ABORT : function, to handle error conditions.
    """
    
    dxinv = 1.0 / delx  # Inverse of delx

    # Check if the total number of data points is even
    if nsp % 2 != 0:
        print(f"Periodic Spline Error: Total number of data points must be even. nsp={nsp} doesn't work.")
        GTC_ABORT()
    
    # Check if the function values at the end points are the same (for periodicity)
    if abs(y[0, 0] - y[0, nsp - 1]) > abs(np.spacing(y[0, 0]) + np.spacing(y[0, nsp - 1])):
        print(f"Periodic Spline Error: Function values at end points must be the same.")
        print(f"y[0] = {y[0, 0]}, y[n] = {y[0, nsp - 1]}")
        print(f"SPACING(y[0, 0]) = {np.spacing(y[0, 0])}, SPACING(y[nsp]) = {np.spacing(y[0, nsp - 1])}")
        GTC_ABORT()

    # Reset higher-order spline coefficients
    y[1:3, :] = 0.0

    # Calculate b1 (the first derivative at the first point)
    for i in range(nsp // 2 - 1):
        y[1, 0] += y[0, 2 * i] - y[0, 2 * i + 1]
    y[1, 0] = y[1, 0] * 2.0 * dxinv

    # Use y[1, nsp - 1] as a buffer to save the periodic b1
    y[1, nsp - 1] = y[1, 0]

    # Calculate the bi (first derivatives) for each point
    for i in range(1, nsp - 1):
        y[1, i] = 2.0 * (y[0, i] - y[0, i - 1]) * dxinv - y[1, i - 1]

    # Calculate the second derivatives for each point
    for i in range(nsp - 1):
        y[2, i] = (y[1, i + 1] - y[1, i]) * 0.5 * dxinv

    # Set the second derivative for the last point to be the same as the first (periodicity)
    y[2, nsp - 1] = y[2, 0]


def mk_splinefunction(nx, delx, f, bcx):
    '''
    constructs a 1d spline of f according to the normal routine,
    returns spline function f(pdum)
    '''
    construct_spline1d(nx, delx, f, bcx)
    if bcx == 0:
        def spl(pdum):
            return spline0(pdum, nx, delx, f)
    elif bcx ==1:
        def spl(pdum):
            return spline1(pdum, nx, delx, f)
    elif bcx ==2:
        print('Periodic spline function has not been added yet')
    return spl
    
    
def construct_spline1d(nx, delx, f, bcx):
    """
    Python translation of the Fortran subroutine construct_spline1d.
    
    Parameters:
    nx : int, number of points
    delx : float, step size
    f : 2D numpy array of shape (3, nx), spline data
    bcx : int, boundary condition flag (0 for construct_spline0, 1 for construct_spline1, 2 for periodic, 3 for no smoothing)
    construct_spline0 : function, the spline constructor for bcx = 0 or 3
    construct_spline1 : function, the spline constructor for bcx = 1
    construct_spline_periodic : function, the spline constructor for bcx = 2 (periodic)
    GTC_ABORT : function to handle error conditions
    """
    
    # Boundary condition handling
    if bcx == 0:
        construct_spline0(0, nx, delx, f)
    elif bcx == 1:
        construct_spline1(nx, delx, f)
    elif bcx == 2:
        construct_spline_periodic(nx, delx, f)
    elif bcx == 3:  # No smoothing
        construct_spline0(3, nx, delx, f)
    else:
        print('Error: Wrong spline boundary condition choice:', bcx)
        GTC_ABORT()/const
    
def construct_spline2d(nx, ny, delx, dely, f, bcx, bcy):
    """
    Python translation of the Fortran subroutine construct_spline2d.
    
    Parameters:
    nx : int, number of points along x-axis
    ny : int, number of points along y-axis
    delx : float, step size along x-axis
    dely : float, step size along y-axis
    f : 3D numpy array of shape (9, nx, ny), spline data
    bcx : int, boundary condition flag for x-axis (1 for no boundary, 2 for periodic)
    bcy : int, boundary condition flag for y-axis (1 for no boundary, 2 for periodic)
    construct_spline1d : function, the 1D spline constructor function
    """
    
    # Initialize temporary arrays for the spline construction
    ddum1 = np.zeros((3, nx))
    ddum2 = np.zeros((3, ny))

    # Periodic condition along the x-axis if bcx == 2
    if bcx == 2:
        f[0, 0, :] = 0.5 * (f[0, 0, :] + f[0, nx-1, :])  # Enforce periodic condition
        f[0, nx-1, :] = f[0, 0, :]

    # Construct splines along the y-axis
    if bcy == 2:
        for j in range(ny - 1):
            ddum1[0, :] = f[0, :, j]
            # Call the 1D spline function (assumed to be passed as an argument)
            construct_spline1d(nx, delx, ddum1, bcx)
            f[0, :, j] = ddum1[0, :]  # Update function values
            f[1, :, j] = ddum1[1, :]  # First derivatives
            f[2, :, j] = ddum1[2, :]  # Second derivatives
        f[0, :, ny-1] = f[0, :, 0]   # Enforce periodic condition
        f[1, :, ny-1] = f[1, :, 0]
        f[2, :, ny-1] = f[2, :, 0]
    else:
        for j in range(ny):
            ddum1[0, :] = f[0, :, j]
            construct_spline1d(nx, delx, ddum1, bcx)
            f[0, :, j] = ddum1[0, :]
            f[1, :, j] = ddum1[1, :]
            f[2, :, j] = ddum1[2, :]

    # Construct splines along the x-axis for each component
    for i in range(nx):
        for s in range(3):
            ddum2[0, :] = f[s, i, :]
            construct_spline1d(ny, dely, ddum2, bcy)
            f[s, i, :] = ddum2[0, :]    # Update function values
            f[s + 3, i, :] = ddum2[1, :]  # First derivatives
            f[s + 6, i, :] = ddum2[2, :]  # Second derivatives


def construct_spline3d(nx, ny, nz, delx, dely, delz, f, bcx, bcy, bcz):
    """
    Python translation of the Fortran subroutine construct_spline3d.
    
    Parameters:
    nx : int, number of points along x-axis
    ny : int, number of points along y-axis
    nz : int, number of points along z-axis
    delx : float, step size along x-axis
    dely : float, step size along y-axis
    delz : float, step size along z-axis
    f : 4D numpy array of shape (27, nx, ny, nz), spline data
    bcx : int, boundary condition flag for x-axis (1 for no boundary, 2 for periodic)
    bcy : int, boundary condition flag for y-axis (1 for no boundary, 2 for periodic)
    bcz : int, boundary condition flag for z-axis (1 for no boundary, 2 for periodic)
    construct_spline2d : function, 2D spline constructor function
    construct_spline1d : function, 1D spline constructor function
    """
    
    # Initialize temporary arrays for the spline construction
    temp1d = np.zeros((3, nz))     # For holding spline data along z-axis
    temp2d = np.zeros((9, nx, ny)) # For holding 2D slices along x-y plane

    # Construct 2D spline on each x-y plane (z direction)
    if bcz == 2:
        for i in range(nz - 1):  # Exclude last point if periodic
            temp2d[0, :, :] = f[0, :, :, i]
            # Call the 2D spline construction
            construct_spline2d(nx, ny, delx, dely, temp2d, bcx, bcy)
            f[0:9, :, :, i] = temp2d[:, :, :]
        # Enforce periodic condition in z
        f[0:9, :, :, nz - 1] = f[0:9, :, :, 0]
    else:
        for i in range(nz):
            temp2d[0, :, :] = f[0, :, :, i]
            construct_spline2d(nx, ny, delx, dely, temp2d, bcx, bcy)
            f[0:9, :, :, i] = temp2d[:, :, :]

    # Generate spline coefficients in the z direction
    for i in range(nx):
        for j in range(ny):
            for s in range(9):
                temp1d[0, :] = f[s, i, j, :]
                # Call the 1D spline constructor for each z-line
                construct_spline1d(nz, delz, temp1d, bcz)
                f[s, i, j, :] = temp1d[0, :]    # Update function values
                f[s + 9, i, j, :] = temp1d[1, :]  # First derivatives
                f[s + 18, i, j, :] = temp1d[2, :]  # Second derivatives

def contour_sp(spline_function, toroidaln, psi=None,zeta = None, psiw = 1, contour_levels=20, ax=None,labels = True,linewidth = 1):
    """
    This code generates a contour plot of a 3D spline function on a toroidal surface.
    The function can plot the spline function in two different modes:
    With a fixed psi value, it plots the function on a toroidal surface with theta/2pi and zeta/2pi as variables.
    With a fixed zeta value, it plots the function on a poloidal surface with x and y as variables, where x and y are related to psi and theta.
    
    Parameters:
    spline_function : function, 3D spline function with inputs (psi, theta, zeta)
    psi : float, fixed psi (radial) value
    toroidaln : int, number of toroidal field periods
    contour_levels : int, number of contour levels in the plot
    ax : matplotlib axis, axis to plot on
    """
    if psi is not None:
        # Create a grid of zeta and theta values
        grid_zeta, grid_theta = np.mgrid[0:2*np.pi/toroidaln:200j, 0:2*np.pi:200j]
        grid_z = np.zeros(np.shape(grid_zeta))
        for i in range(np.shape(grid_zeta)[0]):
            for j in range(np.shape(grid_zeta)[1]):
                grid_z[i, j] = spline_function(psi, grid_theta[i, j], grid_zeta[i, j])

        if ax is None:
            ax = plt.gca()  # Use the current axis if no axis is provided

        # Create the contour plot on the specified axis
        contour = ax.contour(grid_zeta/(2*np.pi), grid_theta/(2*np.pi), grid_z, levels=contour_levels, cmap='binary',linewidths = linewidth)
    if zeta is not None:
        # Create a grid of x and y values
        grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
        grid_z = np.zeros(np.shape(grid_x))
        for i in range(np.shape(grid_y)[0]):
            for j in range(np.shape(grid_x)[1]):
                pdum = psiw*(grid_x[i,j]**2+grid_y[i,j]**2)
                tdum = np.arctan2(grid_y[i,j],grid_x[i,j])% (2 * np.pi)
                if pdum<=psiw:
                    grid_z[i, j] = spline_function(pdum, tdum, zeta)

        if ax is None:
            ax = plt.gca()  # Use the current axis if no axis is provided

        # Create the contour plot on the specified axis
        contour = ax.contour(grid_x, grid_y, grid_z, levels=contour_levels, cmap='binary')#,linewidth = linewidth)

    # Add colorbar
    fig = ax.get_figure()
    if labels:
        fig.colorbar(contour, ax=ax)
    return contour

def contourf_sp(spline_function, toroidaln, psi=None,zeta = None, psiw = 1, contour_levels=20, ax=None,labels = True,linewidth = 1):
    """
    This code generates a contour plot of a 3D spline function on a toroidal surface.
    The function can plot the spline function in two different modes:
    With a fixed psi value, it plots the function on a toroidal surface with theta/2pi and zeta/2pi as variables.
    With a fixed zeta value, it plots the function on a poloidal surface with x and y as variables, where x and y are related to psi and theta.
    
    Parameters:
    spline_function : function, 3D spline function with inputs (psi, theta, zeta)
    psi : float, fixed psi (radial) value
    toroidaln : int, number of toroidal field periods
    contour_levels : int, number of contour levels in the plot
    ax : matplotlib axis, axis to plot on
    """
    if psi is not None:
        # Create a grid of zeta and theta values
        grid_zeta, grid_theta = np.mgrid[0:2*np.pi/toroidaln:200j, 0:2*np.pi:200j]
        grid_z = np.zeros(np.shape(grid_zeta))
        for i in range(np.shape(grid_zeta)[0]):
            for j in range(np.shape(grid_zeta)[1]):
                grid_z[i, j] = spline_function(psi, grid_theta[i, j], grid_zeta[i, j])

        if ax is None:
            ax = plt.gca()  # Use the current axis if no axis is provided

        # Create the contour plot on the specified axis
        contour = ax.contourf(grid_zeta/(2*np.pi), grid_theta/(2*np.pi), grid_z, levels=contour_levels, cmap='binary',linewidths = linewidth)
    if zeta is not None:
        # Create a grid of x and y values
        grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]
        grid_z = np.zeros(np.shape(grid_x))
        for i in range(np.shape(grid_y)[0]):
            for j in range(np.shape(grid_x)[1]):
                pdum = psiw*(grid_x[i,j]**2+grid_y[i,j]**2)
                tdum = np.arctan2(grid_y[i,j],grid_x[i,j])% (2 * np.pi)
                if pdum<=psiw:
                    grid_z[i, j] = spline_function(pdum, tdum, zeta)

        if ax is None:
            ax = plt.gca()  # Use the current axis if no axis is provided

        # Create the contour plot on the specified axis
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap='binary')#,linewidth = linewidth)

    # Add colorbar
    fig = ax.get_figure()
    if labels:
        fig.colorbar(contour, ax=ax)
    return contour