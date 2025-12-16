import numpy as np


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
                            raise ValueError(f"Length of data for key '{key}' does not match 'Pol-Flux'")
                        col = np.asarray(value)
                columns.append(col)
        # Stack columns and write to file
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E')

    def make_profile_lhd(self,path,psi_reff,psiw,r0):
        #this function works similarly to make_profile but is designed for LHD data
        #LHD data is given as with values of reff, so we must convert to psi using psi_p = psi_reff(reff)
        #Ti is given as an array of data while ne, ni, and Ti are expected as functions
        #we then can solve the functions at the points given for ti
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
                    col[i] = self.functions[key](self.data['r'][i])
            else:
                value = self.data[key]
                if np.isscalar(value):
                    col = np.full(n, value)
                else:
                    if len(value) != n:
                        raise ValueError(f"Length of data for key '{key}' does not match 'Pol-Flux'")
                    col = np.asarray(value)
            columns.append(col)
        # Stack columns and write to file
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E', comments='')

    def make_profile_lhd_data(self,path,psi_reff,psiw,r0,points = 100):
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
                    col[i] = self.functions[key](self.data['r'][i])
            else:
                value = self.data[key]
                if np.isscalar(value):
                    col = np.full(n, value)
                    print('Profile for '+key+' is constant value of '+str(value))
                else:
                    if len(value) != n:
                        raise ValueError(f"Length of data for key '{key}' does not match 'Pol-Flux'")
                    col = np.asarray(value)
                    print('Profile for '+key+' added from data')
            columns.append(col)
        # Stack columns and write to file
        data_matrix = np.column_stack(columns)
        header_line = ' '.join(self.data.keys())
        np.savetxt(path, data_matrix, header=header_line, fmt='%.6E', comments='')