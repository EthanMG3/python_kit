#=================================
#====== Stellarator mod ==========
#-----------content---------------
# >> read_spdata
# >> construct_3d_spline
#---------------------------------
# --> Haotian Chen
# --> E-mail: 2101110150@stu.pku.edu.cn
# --> 12/08/2023
# --> edited by ethan in 2025
# --> ethanmg@uci.edu
#=================================

import numpy as np
import math
import spline as sp

def read_spdata(fname,noaxis=1,ndim=9):
    """
    # read spdata.dat for stellarator 
    
    return(sgn,psiw,ped,spdtheta,spdpsi,lsp,lst,torpsi,qpsi,gpsi,cpsi,
    rpsi,ntor,bcn,bsn,xcn,xsn,zcn,zsn,fcn,fsn,ndim,ndim_total,r0,b0)  
    
    # data structure:
     bcn -> ((lsp, lst, ndim_total))
    bsn -> ((lsp, lst, ndim_total))
    xcn -> ((lsp, lst, ndim_total))
    xsn -> ((lsp, lst, ndim_total))
    zcn -> ((lsp, lst, ndim_total))
    zsn -> ((lsp, lst, ndim_total))
    ntor -> (ndim_total)
    fcn -> ((lsp, lst, ndim_total))
    fsn -> ((lsp, lst, ndim_total))
    torpsi -> ((3,lsp))
    cpsi -> ((3,lsp))
    rpsi -> ((3,lsp))
    qpsi -> ((3,lsp))
    gpsi -> ((3,lsp))
    ntor -> (ndim_total)
    """
    with open(fname, 'r') as file:
        line_count = 0
        valid = 1
        # read the first line
        first_line = file.readline().strip()
        line_count += 1
        print(first_line)
        
        # read the second line
        second_line = file.readline().strip().split()
        line_count += 1
        integers = [int(num) for num in second_line]
        # size of equilibrium field data
        lsp,lst=integers
        print(integers)
        
        third_line = file.readline().strip().split()
        line_count += 1
        floats = [float(num) for num in third_line]
        psiw=floats[0]
        print(floats)
        
        
        forth_line=file.readline().strip().split()
        line_count += 1
        integers = [int(num) for num in forth_line]
        [ndim_total,nfp]=integers
        print(integers)
        
        sgn=1
        if psiw<0:
            sgn=-1
        psiw=sgn*psiw
        ped=psiw
        lst=lst+1

        lsp = lsp +noaxis
        if ndim==0:
            ndim = ndim_total
            #then check if the used n mode number exceeds the total read-in mode
            #number
        else: 
            if ndim > ndim_total:
                print("ndim exceeds ndim_total, only", ndim_total,"toroidal harmonics are read and used.")
                ndim = ndim_total
        print("VMEC DIMENSION READ:")
        print("lsp=",lsp,"lst=",lst, "ndim_total=", ndim_total,"psiw=", psiw)
        #generate the array of field data
        
        
        
        
        bcn = np.zeros((lsp, lst, ndim_total))
        bsn = np.zeros((lsp, lst, ndim_total))
        xcn = np.zeros((lsp, lst, ndim_total))
        xsn = np.zeros((lsp, lst, ndim_total))
        zcn = np.zeros((lsp, lst, ndim_total))
        zsn = np.zeros((lsp, lst, ndim_total))
        ntor = np.zeros(ndim_total)
        fcn = np.zeros((lsp, lst, ndim_total))
        fsn = np.zeros((lsp, lst, ndim_total))
        torpsi=np.zeros((3,lsp))
        cpsi=np.zeros((3,lsp))
        rpsi=np.zeros((3,lsp))
        qpsi=np.zeros((3,lsp))
        gpsi=np.zeros((3,lsp))
        ntor=np.zeros(ndim_total)
        
        lines_read=int(math.ceil(ndim_total/6))
        block=[]
        for l in range(0,lines_read):
            line = file.readline().strip()
            line_count += 1
            block.extend([int(num) for num in line.split()])
        ntor[0:ndim_total]=block
        
       
        lines_read=int(math.ceil(lst/4))
        for n in range(1, ndim_total + 1):
            for i in range(1 + noaxis, lsp + 1):
                for data in (bcn,bsn,xcn,xsn,zcn,zsn,fcn,fsn):    
                    block=[]
                    for l in range(0,lines_read):
                        line = file.readline().strip()
                        line_count += 1
                        block.extend([float(num) for num in line.split()])
                    data[i - 1, :, n - 1]= block

        for i in range(1 + noaxis, lsp + 1):
            qpsi[0, i - 1] = float(file.readline())  # q
            gpsi[0, i - 1] = float(file.readline())  # g-current
            cpsi[0, i - 1] = float(file.readline())  # i-current (plasma current)
            rpsi[0, i - 1] = float(file.readline())  # minor radius
            torpsi[0, i - 1] = float(file.readline())  # toroidal flux
        
        fcn = -fcn
        fsn = -fsn
        torpsi = -torpsi
        gpsi = -gpsi
        qpsi = -qpsi
        
 
        fielddir = 0  # setting fielddir
        if (sgn > 0 and torpsi[0, lsp-1] > 0):
            fielddir = 0
        elif (sgn > 0 and torpsi[0, lsp-1] < 0):
            fielddir = 1
        elif (sgn < 0 and torpsi[0, lsp-1] < 0):
            fielddir = 2
        elif (sgn < 0 and torpsi[0, lsp-1] > 0):
            fielddir = 3

        print("In zeta out system: sgn(psip)=", sgn, "torpsi(1,lsp)=", torpsi[0, lsp-1], "gpsi(1,1)=", gpsi[0, 0], 'fielddir=', fielddir)

        if fielddir in (2, 3):
            fcn = -fcn
            fsn = -fsn
            torpsi = -torpsi
            gpsi = -gpsi
            cpsi = -cpsi


        # normalized to GTC unit
        x00 = np.sum(xcn[1 + noaxis, :, 0]) / lst
        b00 = np.sum(bcn[1 + noaxis, :, 0]) / lst

        print("Using VMEC: R0=", x00, "B0=", b00, 'ndim=', ndim_total)

        r0 = x00 * 100.0
        b0 = b00 * 10000.0

        # normalized to GTC unit
        psiw = psiw / (b00 * x00 * x00)
        ped = ped / (b00 * x00 * x00)
        spdtheta = 2.0 * math.pi / (lst - 1)
        spdpsi = ped / (lsp - 1)
        torpsi = torpsi / (b00 * x00 * x00)
        gpsi = gpsi / (b00 * x00)
        cpsi = cpsi / (b00 * x00)
        bcn = bcn / b00
        bsn = bsn / b00
        xcn = xcn / x00
        xsn = xsn / x00
        zcn = zcn / x00
        zsn = zsn / x00
        rpsi = rpsi / x00
        return(sgn,psiw,ped,spdtheta,spdpsi,lsp,lst,torpsi,qpsi,gpsi,cpsi,rpsi,ntor,bcn,bsn,xcn,xsn,zcn,zsn,fcn,fsn,ndim,ndim_total,r0,b0,nfp)  
    
def construct_3d_spline(sgn,psiw,ntor,lsp,lst,lszeta,toroidaln,ndim,noaxis,ndim_total,ycn,ysn,mk_spline=True):
    '''
    sgn,ntor,lsp,lst,ndim,noaxis,ndim_total same as output by read_spdata
    
    lszeta=mtoroidal*nzsp_sec+1
    
    nzsp_sec,toroidaln in gtc.in
    
    ycn,ysn can be replaced by bcn,bsn  xcn,xsn  zcn,zsn
    '''
    sptmp3d=np.zeros((27,lsp,lst,lszeta))
    spdpsi = psiw/(lsp-1)
    spdtheta = 2.0*np.pi/(lst-1)
    spdzeta = 2.0*np.pi/((lszeta-1)*toroidaln)

    if sgn > 0:
        for k in range(lszeta):
            zdum = -float(k) * spdzeta
            for n in range(ndim):
                cosnz = np.cos(float(ntor[n])  * zdum)
                sinnz = np.sin(float(ntor[n])  * zdum)
                for j in range(lst):
                    for i in range(noaxis, lsp):
                        sptmp3d[0, i , j, k] += ycn[i , j , n] * cosnz + ysn[i, j , n] * sinnz
    else:
        for k in range(lszeta):
            zdum = float(k) * spdzeta
            for n in range( ndim ):
                cosnz = np.cos(float(ntor[n])  * zdum)
                sinnz = np.sin(float(ntor[n])  * zdum)
                for j in range(lst):
                    for i in range(noaxis, lsp):
                        sptmp3d[0, i , j , k ] += ycn[i , lst - j-1, n] * cosnz + ysn[i, lst - j-1, n ] * sinnz   
    
    for z in range(lszeta):
        sptmp3d[0,0,:,z] = np.average(sptmp3d[0,1,:,z]) 
    print('Completed mapping to boozer coordinates')
    if mk_spline:
        bcx = 1
        bcy = 2
        bcz = 2
        print('begin constructing spline')
        sp.construct_spline3d(lsp, lst, lszeta, spdpsi, spdtheta, spdzeta,sptmp3d, bcx, bcy, bcz)
        
    
        def spline_function(psi,theta,zeta):
            bcx =1
            bcy=2
            bcz=2
            return sp.spline3d(psi, theta, zeta, 0, lsp, lst, lszeta, spdpsi, spdtheta, spdzeta,sptmp3d, bcx, bcy, bcz)
    
        return sptmp3d,spline_function
    
    return sptmp3d


class Stellarator:
    def __init__(self, filepath, noaxis=1, ndim=9, mtoroidal = None, nzsp_sec=None, toroidaln=1,make3dspline=True,make1dspline=True):

        # Load equilibrium data
        self.filepath = filepath
        self.noaxis = noaxis
        self.ndim = ndim
        (self.sgn, self.psiw, self.ped, self.spdtheta, self.spdpsi,
         self.lsp, self.lst, self.torpsi, self.qpsi, self.gpsi,
         self.cpsi, self.rpsi, self.ntor, self.bcn, self.bsn,
         self.xcn, self.xsn, self.zcn, self.zsn, self.fcn, self.fsn,
         self.ndim, self.ndim_total, self.r0, self.b0, self.nfp) = read_spdata(
            filepath, noaxis=noaxis, ndim=ndim
        )

        self.splines = {}
        # Optional spline construction
        if mtoroidal is not None and nzsp_sec is not None:
            self.lszeta = mtoroidal*nzsp_sec+1
            self.spdzeta = 2*np.pi/toroidaln/(self.lszeta-1)
        if toroidaln is not None:
            self.toroidaln = toroidaln
        if make3dspline:
            self._make_spline_functions3d()
        if make1dspline:
            self._make_spline_functions1d()

    def _make_spline_functions3d(self):
        
        for label, ycn, ysn in [
            ('B', self.bcn, self.bsn),
            ('X', self.xcn, self.xsn),
            ('Z', self.zcn, self.zsn),
            ('F', self.fcn, self.fsn)
        ]:
            sptmp3d, spline_fn = construct_3d_spline(
                self.sgn, self.psiw, self.ntor, self.lsp, self.lst,
                self.lszeta, self.toroidaln, self.ndim, self.noaxis,
                self.ndim_total, ycn, ysn, mk_spline=True
            )
            self.splines[label] = {
                'data': sptmp3d,
                'function': spline_fn
            }
        print('Spline functions for B, X, Z, and F constructed')

    def _make_spline_functions1d(self):

        spline_fn = sp.mk_splinefunction(self.lsp,self.spdpsi,self.rpsi,1)

        self.splines['r_eff'] = {
                'function': spline_fn
            }

        for label, sp0d in [
            ('q', self.qpsi),
            ('g', self.gpsi),
            ('c', self.cpsi),
            ('psi_t', self.torpsi)
        ]:
            spline_fn = sp.mk_splinefunction(self.lsp,self.spdpsi,sp0d,0)

            self.splines[label] = {
                'function': spline_fn
            }
        print('Spline functions for r, q, g, c, $\psi_t$ constructed')
     

    def get_spline_function(self, label):
        return self.splines[label]['function'] if label in self.splines else None
