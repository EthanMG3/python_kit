import stellarator as stl
import spline as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import tracking as Tra
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time


interppoints = 100
gridpoints = 120j
# Record the start time
start_time = time.time()

#read stellarator data
case_path = '/home/emgreen/simulations/stellarator/183281/uniform/no_Er/'
fname="spdata.dat"
(sgn, psiw, ped, spdtheta, spdpsi, lsp, lst, torpsi, qpsi, gpsi, cpsi, rpsi,
 ntor, bcn, bsn, xcn, xsn, zcn, zsn, fcn, fsn, ndim, ndim_total, r0, b0, nfp) = stl.read_spdata(
    case_path + fname, 0, 9
)

#produce stellarator splines
nzsp_sec=7
mtoroidal=15
lszeta=mtoroidal*nzsp_sec+1
toroidaln=5
spdzeta = 2*np.pi/toroidaln/(lszeta-1)
ndim=14
noaxis=1
ndim_total=14
spx3d,spx=stl.construct_3d_spline(sgn,psiw,ntor,lsp,lst,lszeta,toroidaln,ndim,noaxis,ndim_total,xcn,xsn)
spz3d,spz=stl.construct_3d_spline(sgn,psiw,ntor,lsp,lst,lszeta,toroidaln,ndim,noaxis,ndim_total,zcn,zsn)
spb3d,spb=stl.construct_3d_spline(sgn,psiw,ntor,lsp,lst,lszeta,toroidaln,ndim,noaxis,ndim_total,bcn,bsn)
spf3d,spf=stl.construct_3d_spline(sgn,psiw,ntor,lsp,lst,lszeta,toroidaln,ndim,noaxis,ndim_total,fcn,fsn)
del xcn,xsn,zcn,zsn,bcn,bsn,fcn,fsn
del spz3d,spx3d,spb3d,spf3d
t1 = time.time()
def cartesian_boozer(psi,theta,zeta):
    #returns cartesian coordinates of point in boozer coordinates
    #requires all 4 3d sspline functions
    r = spx(psi,theta,zeta)
    z = spz(psi,theta,zeta)
    f = -spf(psi,theta,zeta)-zeta
    x = r*np.cos(f)
    y = r*np.sin(f)
    return x,y,z

# Inverse function to find (psi, theta, zeta) given (x, y, z)
def inverse_cartesian(x_target, y_target, z_target):
    zeta_guess = min(np.arctan(-y_target/x_target),np.pi*2/5)
    if x_target**2+y_target**2<1:
        theta_guess = np.arctan(z_target/((x_target-np.cos(zeta_guess))**2+(y_target+np.sin(zeta_guess))**2))
        theta_guess = np.pi-theta_guess
    else:
        theta_guess = np.arctan(z_target/((x_target-np.cos(zeta_guess))**2+(y_target+np.sin(zeta_guess))**2))
        if theta_guess<0:
            theta_guess = 2*np.pi+theta_guess
    initial_guess=(psiw, theta_guess,zeta_guess)
    # Objective function to minimize the distance between target and calculated (x, y, z)
    def objective(ptz):
        psi, theta, zeta = ptz
        x, y, z = cartesian_boozer(psi, theta, zeta)
        return np.linalg.norm([x - x_target, y - y_target, z - z_target])
    xt,yt,zt = cartesian_boozer(psiw, theta_guess, zeta_guess)
    r = (xt-np.cos(zeta_guess))**2+(yt+np.sin(zeta_guess))**2+zt**2
    r_target = (x_target-np.cos(zeta_guess))**2+(y_target+np.sin(zeta_guess))**2+z_target**2
    if r_target>r:
        return [psiw+1,-1,-1]
    # Minimize the objective function to find (psi, theta, zeta)
    result = minimize(objective, initial_guess, bounds=[(0, psiw), (0, 2 * np.pi), (0, 2 * np.pi/5)])
    xt,yt,zt = cartesian_boozer(result.x[0],result.x[1],result.x[2])
    if ((xt-x_target)**2+(yt-y_target)**2+(zt-z_target)**2)>0.001:
        return [psiw+1,-1,-1]
    return result.x if result.success else [psiw+1,-1,-1]  # Returns (psi, theta, zeta) or None if failed


# Define ranges for psi, theta, and zeta
psi_vals = np.linspace(psiw*0.1, 0.9*psiw, interppoints)       # Adjust psi_max as needed
theta_vals = np.linspace(0, 2 * np.pi, interppoints)
zeta_vals = np.linspace(0, 2 * np.pi/5, interppoints)

# Create a meshgrid for psi, theta, and zeta
psi, theta, zeta = np.meshgrid(psi_vals, theta_vals, zeta_vals)

# Flatten the grid arrays to pass through functions
psi_flat = psi.flatten()
theta_flat = theta.flatten()
zeta_flat = zeta.flatten()

# Prepare arrays for Cartesian coordinates and magnetic field values
x = np.zeros(len(psi_flat))
y = np.zeros(len(psi_flat))
z = np.zeros(len(psi_flat))
b_field = np.zeros(len(psi_flat))

# Convert (psi, theta, zeta) to Cartesian coordinates and compute B field
for i in range(len(psi_flat)):
    x[i], y[i], z[i] = cartesian_boozer(psi_flat[i], theta_flat[i], zeta_flat[i])
    b_field[i] = spb(psi_flat[i], theta_flat[i], zeta_flat[i])

del psi,theta,zeta,psi_flat,theta_flat,zeta_flat

# Data coordinates
points = np.array( (z.flatten(), x.flatten(), y.flatten()) ).T
# Data values @ above coords
values = b_field.flatten()

# Grid to interp to
Z, X, Y = np.mgrid[z.min():z.max():gridpoints, x.min():x.max():gridpoints, y.min():y.max():gridpoints]
# New data values on interp grid
newdata = griddata( points, values, (Z,X,Y) )
# Parallelized function
def check_and_mask(index):
    pdum = inverse_cartesian(X[index], Y[index], Z[index])
    if pdum[0] >= psiw:
        return np.nan
    else:
        return newdata[index]
    
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
newdata = newdata.flatten()
# Run in parallel
newdata = np.array(Parallel(n_jobs=-1)(delayed(check_and_mask)(i) for i in range(len(newdata))))
t2 = time.time()
fig = go.Figure(data=[
            go.Isosurface(x=X,y=Y,z=Z,
                       value=newdata.flatten(),
                       opacity=0.3,
                       isomin=0.5,
                       isomax=1.5,
                       surface_count=17,
                       colorscale='viridis',
                       showscale=True,
                       caps=dict(x_show=False, y_show=False)
                       ),])


# Save as HTML
fig.write_html("b3dsurf.html",full_html=False)

'''fig = go.Figure(data=[
            go.Volume(x=X,y=Y,z=Z,
                       value=newdata.flatten(),
                       opacity=0.3,
                       isomin=0.5,
                       isomax=1.5,
                       surface_count=17,
                       colorscale='viridis',
                       showscale=True,
                       caps=dict(x_show=False, y_show=False)
                       ),])


# Save as HTML
fig.write_html("b3dvol.html",full_html=False)
'''
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = t1 - start_time
print(f"Loading and Spline creation: {elapsed_time:.2f} seconds")
elapsed_time = t2 - t1
print(f"Interpolation: {elapsed_time:.2f} seconds")
elapsed_time = end_time - t1
print(f"Figure production: {elapsed_time:.2f} seconds")
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
print('Interpolation points: ',str(interppoints),str(gridpoints) )