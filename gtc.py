#=================================
#======= gtcout Module =========
#-----------content---------------
# >> read
# >> gridshow
# >> profileshow
#---------------------------------
# --> Yangyang Yu
# --> E-mail: yangyy3@uci.edu
# --> 2023/07/03
#=================================

import re
import numpy as np
import matplotlib.pyplot as plt

def read(file_path): 
  """
  This reads gtc.out files into python.
  return (physical_parameters, radial_grid, radial_profile)
  """

  key_line_1 = "&PHYSICAL_PARAMETERS"
  key_line_2 = "i,  rg/a,  psi/ped,   q,   rg_sp/rg - 1,  dtorpsi/q "
  key_line_3 = "i  ,ne   ,Te  ,Ti  ,meshte  ,meshti   ,meshne  ,meshni  "
  key_line_4 = "i,   mtheta   "

  pattern = r'(\w+)\s*=\s*([\d.Ee+-]+)'

  # Physical parameters
  physical_parameters = {}
  start_processing1 = False

  # Radial grid
  list1 = []
  start_processing2 = False

  # Key profiles on radial mesh
  list2 = []
  start_processing3 = False

  # Read gtc.out
  with open(file_path, "r") as file:
      for line in file:
          if start_processing1:
              # Find all matches in the string
              matches = re.findall(pattern, line)

              # Process the matches
              for match in matches:
                  name = match[0]
                  number = float(match[1])
                  physical_parameters[name] = number    
                
          if key_line_1 in line:
              start_processing1 = True
            
          if start_processing2:
              matches = line.split()
              for match in matches:
                  try:
                      number = float(match)
                      list1.append(number)
                  except ValueError:
                      start_processing2=False
                
          if key_line_2 in line:
              start_processing1 = False
              start_processing2 = True
        
          if start_processing3:
              matches = line.split()
              for match in matches:
                  try:
                      number = float(match)
                      list2.append(number)
                  except ValueError:
                      start_processing3=False
                
          if key_line_3 in line:
              start_processing2 = False
              start_processing3 = True
        
          if key_line_4 in line:
              break

  # Reshape list1 to get radial grid information. psi: 1, mpsi-1
  mpsi=int(len(list1)/6+1)
  radial_grid=np.zeros((mpsi-1,6),dtype=float)
  for i in range(0,mpsi-1):
      for j in range(0,6):
          radial_grid[i,j]=list1[j+i*6]

  # Reshape list2 to get key profiles on radial mesh. psi: 0, mpsi
  mpsi=int(len(list2)/8-1)
  radial_profile=np.zeros((mpsi+1,8),dtype=float)
  for i in range(0,mpsi+1):
      for j in range(0,8):
          radial_profile[i,j]=list2[j+i*8]

  return(physical_parameters,radial_grid,radial_profile)


###################################
## plot 
###################################
def gridshow(radial_grid, kind = 3, savefig = 0):
    """
    show poloidal cross section
    # kind =
        1: rg/a, 2: psi/ped, 3: q (default)
        4: rg_sp/rg - 1, 5: dtorpsi/q
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    
    fig, ax= plt.subplots(figsize=(5,3), dpi = 120)
    if kind==1:
        x=radial_grid[:,0]
        y=radial_grid[:,1]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='rg/a',title='')
    elif kind==2:
        x=radial_grid[:,1]
        y=radial_grid[:,2]
        ax.plot(x,y)
        ax.set(xlabel='rg/a', ylabel='psi/ped',title='')
    elif kind==3:
        x=radial_grid[:,1]
        y=radial_grid[:,3]
        ax.plot(x,y)
        ax.set(xlabel='rg/a', ylabel='q',title='')
    elif kind==4:
        x=radial_grid[:,1]
        y=radial_grid[:,4]
        ax.plot(x,y)
        ax.set(xlabel='rg/a', ylabel='rg_sp/rg - 1',title='')
    elif kind==5:
        x=radial_grid[:,1]
        y=radial_grid[:,5]
        ax.plot(x,y)
        ax.set(xlabel='rg/a', ylabel='dtorpsi/q',title='')
         
    ax.grid()       
    if savefig:
        fig.savefig("radial_grid")
    return


def profileshow(radial_profile, kind = 1, savefig = 0):
    """
    show poloidal cross section
    # kind =
        1: ne (default), 2: Te, 3: Ti, 4: meshte, 5: meshti, 6: meshne, 7: meshni
    # savefig
        0: don't save figure (default)
        1: save figure
    """
    
    fig, ax= plt.subplots(figsize=(5,3), dpi = 120)
    if kind==1:
        x=radial_profile[:,0]
        y=radial_profile[:,1]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='ne',title='')
    elif kind==2:
        x=radial_profile[:,0]
        y=radial_profile[:,2]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='Te',title='')
    elif kind==3:
        x=radial_profile[:,0]
        y=radial_profile[:,3]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='Ti',title='')
    elif kind==4:
        x=radial_profile[:,0]
        y=radial_profile[:,4]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='meshte',title='')
    elif kind==5:
        x=radial_profile[:,0]
        y=radial_profile[:,5]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='meshti',title='')
    elif kind==6:
        x=radial_profile[:,0]
        y=radial_profile[:,6]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='meshne',title='')
    elif kind==7:
        x=radial_profile[:,0]
        y=radial_profile[:,7]
        ax.plot(x,y)
        ax.set(xlabel='# of psi flux', ylabel='meshni',title='')
         
    ax.grid()       
    if savefig:
        fig.savefig("radial_profile")
    return
