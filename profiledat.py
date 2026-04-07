import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class Profile:
    def __init__(self):
        self.data = {'Pol-Flux':0, 'x':0, 'r':0, 'R':0, 'R+r':0, 'Te':0, 'ne':0, 'Ti':0, 'Zeff':0, 
                        'omega-tor':0, 'Er':0, 'ni':0, 'nimp':0, 'nf':0, 'Tf':0}
        self.functions = {}

    def add_data(self, key, values):
        self.data[key] = values

    def add_function(self, key, function):
        self.functions[key] = function

    def make_profile(self,path,points = 0):
        if points == 0:
            # Check that Pol-Flux is present and an array
            if 'Pol-Flux' not in self.data:
                raise ValueError("Missing required key 'Pol-Flux'")
            pol_flux = np.array(self.data['Pol-Flux'])
            n = len(pol_flux)

            # Build columns
            columns = []
            for key in self.data.keys():
                value = self.data[key]
                if np.isscalar(value):
                    col = np.full(n, value)
                else:
                    value = np.asarray(value)
                    if len(value) != n:
                        raise ValueError(f"Length of data for key '{key}' does not match 'Pol-Flux'")
                    col = value
                columns.append(col)
        else:
            pol_flux = np.linspace(0,1,points)**2
            n = len(pol_flux)
            columns = []
            for key in self.data.keys():
                if key == 'Pol-Flux':
                    columns.append(pol_flux)
                    continue
                col = np.zeros(n)
                if key in self.functions:
                    for i in range(n):
                        # Evaluate function
                        col[i] = self.functions[key](pol_flux[i])
                else:
                    value = self.data[key]
                    if np.isscalar(value):
                        col = np.full(n, value)
                    else:
                        if len(value) != n:
                            raise ValueError(f"Length of data for key '{key}' does not match \"Pol-Flux\"")
                        col = np.asarray(value)
                columns.append(col)
        # Stack columns and write to file
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E')

    def make_profile_lhd(self,path,psi_reff,psiw,r0,a,points = 100):
        #this function works similarly to make_profile but is designed for LHD data
        #LHD data is given as with values of reff, so we must convert to psi using psi_p = psi_reff(reff)
        #Ti is given as an array of data while ne, ni, and Ti are expected as functions
        #we then can solve the functions at the points given for ti
        self.data['r'] = np.linspace(self.data['r'][0],a,points)
        self.pol_flux = psi_reff(self.data['r']/r0)/psiw
        n = len(self.data['r'])
        columns = []
        for key in self.data.keys():
            if key == 'Pol-Flux':
                columns.append(self.pol_flux)
                continue
            col = np.zeros(n)
            if key in self.functions:
                for i in range(n):
                    # Evaluate function as a function of reff
                    col[i] = self.functions[key](self.data['r'][i]/a)
            else:
                value = self.data[key]
                if np.isscalar(value):
                    col = np.full(n, value)
                else:
                    if len(value) != n:
                        raise ValueError(f"Length of data for key '{key}' does not match \"Pol-Flux\"")
                    col = np.asarray(value)
            columns.append(col)
        # Stack columns and write to file
        print(self.data['r'])
        print('r0 = ',a)
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E', comments='')

    def make_profile_lhd_data(self,path,psi_reff,psiw,r0,a,points = 100):
        #this function works similarly to make_profile but is designed for LHD data
        #LHD data is given as with values of reff, so we must convert to psi using psi_p = psi_reff(reff)
        #Ti is given as an array of data while ne, ni, and Ti are expected as functions
        #we then can solve the functions at the points given for ti
        self.data['r'] = np.linspace(self.data['r'][0],self.data['r'][-1],points)
        self.pol_flux = psi_reff(self.data['r']/r0)/psiw
        n = len(self.data['r'])
        columns = []
        for key in self.data.keys():
            if key == 'Pol-Flux':
                columns.append(self.pol_flux)
                continue
            col = np.zeros(n)
            if key in self.functions:
                print('Profile for '+key+' created from function')
                for i in range(n):
                    # Evaluate function as a function of reff
                    col[i] = self.functions[key](self.data['r'][i]/a)
            else:
                value = self.data[key]
                if np.isscalar(value):
                    col = np.full(n, value)
                    print('Profile for '+key+' is constant value of '+str(value))
                else:
                    if len(value) != n:
                        raise ValueError(f"Length of data for key '{key}' does not match \"Pol-Flux\"")
                    col = np.asarray(value)
                    print('Profile for '+key+' added from data')
            columns.append(col)
        # Stack columns and write to file
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E', comments='')

    def make_runspec(
        self,
        path,
        stellarator,
        psiN_wish=41,
        ne_key='ne',
        te_key='Te',
        ti_key='Ti',
        ne_scale=1e20,
        t_scale=1e3,
        inversion_points=400,
        verbose=False
    ):
        """
        Build a SFINCS-style runspec file from profile functions and a stellarator mapping.

        Coordinate flow:
            psiN = psi_t(psi) / psi_t(psiw)
            x_profile = r / r_edge, with r = rpsi(psi), r_edge = rpsi(psiw)

        The profile functions (ne/Te/Ti) are evaluated at x_profile.
        """
        if not hasattr(stellarator, 'psi_t'):
            stellarator._make_spline_functions1d()


        psiN = np.linspace(0.0, 1.0, int(psiN_wish))

        if psiN.ndim != 1:
            raise ValueError("psiN_wish must be a 1D array or an integer number of points")
        if psiN.size < 2:
            raise ValueError("psiN_wish must contain at least 2 points")
        if np.any(np.diff(psiN) <= 0):
            raise ValueError("psiN_wish must be strictly increasing")

        if ne_key not in self.functions:
            raise ValueError(f"Missing required function '{ne_key}' in self.functions")
        if te_key not in self.functions:
            raise ValueError(f"Missing required function '{te_key}' in self.functions")

        psiw = stellarator.psiw
        psi_t = stellarator.get_spline_function('psi_t')
        psi_tw = float(psi_t(float(psiw)))

        psi_from_psit = invert_function(
            psi_t,
            xstart=0.0,
            xstop=psiw,
            npoints=inversion_points,
            verbose=verbose,
            breakpoint=100000
        )
        # Inverse spline is only defined on [psi_t_axis, psi_t_edge]. Evaluating outside
        # (e.g. at 0 when psi_t(0) != 0) causes CubicSpline extrapolation → wrong edge values.
        psi_t_wish = np.clip(psiN * psi_tw, 0, psi_tw)
        psi_wish = np.asarray(psi_from_psit(psi_t_wish), dtype=float)
        # Guard against inverse-spline overshoot near axis/edge.
        psi_wish = np.clip(psi_wish, 0.0, float(psiw))

        x = np.linspace(0.0,psi_tw,100)
        plt.plot(x,psi_from_psit(x),label='inverse psi_t')
        plt.plot(psi_t_wish,psi_wish,label='psi_wish')
        plt.legend()
        plt.title('psi vs psi_t')
        plt.show()
        if not np.all(np.isfinite(psi_wish)):
            good = np.isfinite(psi_wish)
            if np.count_nonzero(good) < 2:
                raise ValueError("Could not construct finite psi values from inverse psi_t mapping")
            psi_wish[~good] = np.interp(psiN[~good], psiN[good], psi_wish[good])

        rpsi = stellarator.get_spline_function('r_eff')
        r_wish = np.asarray([rpsi(float(psi_val)) for psi_val in psi_wish], dtype=float)
        plt.plot(psi_t_wish,r_wish)
        plt.title('r vs psi_t')
        plt.show()
        if not np.all(np.isfinite(r_wish)):
            good = np.isfinite(r_wish)
            if np.count_nonzero(good) < 2:
                raise ValueError("Could not construct finite r values from r_eff mapping")
            r_wish[~good] = np.interp(psiN[~good], psiN[good], r_wish[good])
        r_edge = float(rpsi(float(psiw)))
        if np.isclose(r_edge, 0.0):
            raise ValueError("rpsi(psiw) is zero; cannot normalize radius")
        x_profile = r_wish / r_edge

        ne_values = np.asarray([10**6*self.functions[ne_key](xi) for xi in x_profile], dtype=float) #density profiles used in gtc in cm^-3
        te_values = np.asarray([self.functions[te_key](xi) for xi in x_profile], dtype=float)

        if ti_key in self.functions:
            ti_values = np.asarray([self.functions[ti_key](xi) for xi in x_profile], dtype=float)
        else:
            ti_values = te_values.copy()

        nHats_1 = ne_values / ne_scale
        nHats_2 = nHats_1.copy()
        THats_1 = te_values / t_scale
        THats_2 = ti_values / t_scale

        dnHatdpsiNs_1 = np.gradient(nHats_1, psiN)
        dnHatdpsiNs_2 = dnHatdpsiNs_1.copy()
        dTHatdpsiNs_1 = np.gradient(THats_1, psiN)
        dTHatdpsiNs_2 = np.gradient(THats_2, psiN)

        nHats = [nHats_1, nHats_2]
        dnHatdpsiNs = [dnHatdpsiNs_1, dnHatdpsiNs_2]
        THats = [THats_1, THats_2]
        dTHatdpsiNs = [dTHatdpsiNs_1, dTHatdpsiNs_2]
        species_labels = [f"species {i+1}" for i in range(len(nHats))]

        plt.figure()
        for nhat, label in zip(nHats, species_labels):
            plt.plot(psiN[1:-1], nhat[1:-1], label=label)
        plt.xlabel(r"$\psi_N$")
        plt.ylabel(r"$\hat{n}$")
        plt.title("Density vs $\\psi_N$")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.figure()
        for dnhat, label in zip(dnHatdpsiNs, species_labels):
            plt.plot(psiN[1:-1], dnhat[1:-1], label=label)
        plt.xlabel(r"$\psi_N$")
        plt.ylabel(r"$d\hat{n}/d\psi_N$")
        plt.title("Density derivative vs $\\psi_N$")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.figure()
        for that, label in zip(THats, species_labels):
            plt.plot(psiN[1:-1], that[1:-1], label=label)
        plt.xlabel(r"$\psi_N$")
        plt.ylabel(r"$\hat{T}$")
        plt.title("Temperature vs $\\psi_N$")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.figure()
        for dthat, label in zip(dTHatdpsiNs, species_labels):
            plt.plot(psiN[1:-1], dthat[1:-1], label=label)
        plt.xlabel(r"$\psi_N$")
        plt.ylabel(r"$d\hat{T}/d\psi_N$")
        plt.title("Temperature derivative vs $\\psi_N$")
        plt.grid(True, alpha=0.3)
        plt.legend()

        data_matrix = np.column_stack([
            psiN,
            nHats_1, nHats_2,
            THats_1, THats_2,
            dnHatdpsiNs_1, dnHatdpsiNs_2,
            dTHatdpsiNs_1, dTHatdpsiNs_2
        ])[1:-1]
        header_line = (
            "psiN_wish nHats_1 nHats_2 THats_1 THats_2 "
            "dnHatdpsiNs_1 dnHatdpsiNs_2 dTHatdpsiNs_1 dTHatdpsiNs_2"
        )
        output_path = path
        if os.path.isdir(path):
            output_path = os.path.join(path, 'runspec.dat')
        np.savetxt(output_path, data_matrix, header=header_line, fmt='%.16g', comments='! ')

def closest_time(df, time):
    t = df.dropna().index.values
    t_closest = t[np.abs(time - t).argmin()]
    return t_closest

def return_value(df, time,value,shot):
    t_closest = closest_time(df, time)
    return df[value,shot].loc[t_closest]

def te_profile(df, time, shot):
    t_closest = closest_time(df, time)
    print('Closest time: ',t_closest)
    te0 = df['cte0',shot].loc[t_closest]
    te2 = df['cte2',shot].loc[t_closest]
    te4 = df['cte4',shot].loc[t_closest]
    te6 = df['cte6',shot].loc[t_closest]
    def function(x):
        return 1000*(te0+te2*x**2+te4*x**4+te6*x**6)
    return function

def ne_profile(df, time, shot):
    #produce density profile in cm^-3 as a function of normalized radius
    t_closest = closest_time(df, time)
    print('Closest time: ',t_closest)
    ne_df = df.loc[t_closest].filter(regex=r'cne\d').filter(like=(shot)).droplevel('shots')
    def function(x):
        return 1e13*sum([val*(x)**int(key[-1]) for key, val in ne_df.items()])
    return function


def ni_profile(df, time, shot,species):
    #produce density profile in cm^-3 as a function of normalized radius
    t_closest = closest_time(df, time)
    print('Closest time: ',t_closest)
    if species == 'h':
        ne_df = df.loc[t_closest].filter(regex=r'nh\d').filter(like=(shot)).droplevel('shots')
    elif species == 'd':
        ne_df = df.loc[t_closest].filter(regex=r'nd\d').filter(like=(shot)).droplevel('shots')
    elif species == 'he':
        ne_df = df.loc[t_closest].filter(regex=r'nhe\d').filter(like=(shot)).droplevel('shots')
    elif species == 'c':
        ne_df = df.loc[t_closest].filter(regex=r'nc\d').filter(like=(shot)).droplevel('shots')
        
    def function(x):
        return 1e13*sum([val*(x)**int(key[-1]) for key, val in ne_df.items()])
    return function

def ti_profile_from_data(reff, ti, max_order=6):
    """
    Fit Ti(reff) with an even polynomial:
        Ti(r) ≈ a0 + a2 r^2 + a4 r^4 + a6 r^6  (for max_order=6)
    """
    reff = np.asarray(reff)
    ti   = np.asarray(ti)

    # polynomial in r^2 → degrees 0,2,4,6 in r
    deg = max_order // 2
    x2 = reff**2
    coeffs = np.polyfit(x2, ti, deg=deg)  # highest power first

    def function(x):
        return np.polyval(coeffs, x**2)

    plt.plot(reff,function(reff),label='fit')
    plt.scatter(reff,ti,label='data',color = 'orange')
    plt.legend()
    plt.show()
    return function

def prepare_lhd_profile(profile,time,shot,a,cxs_map,tswpe,cxs7,species='h',zeff=1.0):
    t_closest = closest_time(cxs_map,time)
    #print(t_closest)
    reff = np.asarray(cxs_map.loc[t_closest]['reff',shot][(0<cxs_map.loc[t_closest]['reff',shot])&(cxs_map.loc[t_closest]['reff',shot]<a)],dtype=float)
    ti   = 1000*np.asarray(cxs_map.loc[t_closest]['ti',shot][(0<cxs_map.loc[t_closest]['reff',shot])&(cxs_map.loc[t_closest]['reff',shot]<a)],dtype=float)
    # extrapolate ion temperature to axis assuming quadratic profile with 0 slope on axis
    cr2 = (ti[0]-ti[1])/(reff[0]**2-reff[1]**2)
    ti0 = ti[0]-cr2*reff[0]**2
    reff = np.insert(reff,0,0)
    ti   = np.insert(ti,0,ti0)

    profile.add_data("r", reff)

    # Ti as an even polynomial profile built from data
    profile.add_function("Ti", ti_profile_from_data(reff/a, ti, max_order=6))

    profile.add_function("Te", te_profile(tswpe,time,shot))
    profile.add_function("ne", ne_profile(tswpe,time,shot))
    # I can add code later to get zeff and the corresponding ion density profiles here
    profile.add_function("ni", ni_profile(cxs7,time,shot,species))
    profile.add_data("Zeff", zeff)

def invert_function(func,xstart=0.0,xstop=1.0,npoints=100,verbose=False,breakpoint=40000):
    granularity = 10e-10
    ystart = func(xstart)
    ystop = func(xstop)
    if verbose:
        print(ystart,ystop)
    convergence_condition = (ystop-ystart)/(100*npoints)
    if verbose:
        print(convergence_condition)
    y = np.linspace(ystart,ystop,npoints)
    x = np.zeros(len(y))
    for i,y_val in enumerate(y):
        if i==0:
            x[i] = xstart
            continue
        if i==npoints-1:
            x[i] = xstop
            continue    
        #gradient decent search
        xguess = 0.5*(xstart+xstop)
        step = 0
        converged = False
        step_size = 0.2*(xstop-xstart)
        if verbose:
            print(y_val)
        while not converged:
            #print(xguess,y_val-func(xguess))
            step += 1
            dfdx = (func(xguess+granularity)-func(xguess-granularity))/(2*granularity)
            step_size = step_size*0.9999
            dJdx = -2*(y_val-func(xguess))*dfdx
            #print(dfdx,dJdx,step_size)
            xguess = max(xstart+granularity,min(xstop-granularity,xguess - (step_size)*dJdx))
            if np.abs(y_val-func(xguess))<convergence_condition:
                converged = True
                if verbose:
                    print(func(xguess), 'converged in',step,'steps')
            if step>breakpoint:
                print('did not converge',func(xguess),' to ',y_val,' in ',step,'steps')
                break
        x[i] = xguess
    inv_func = CubicSpline(y,x)
    return inv_func