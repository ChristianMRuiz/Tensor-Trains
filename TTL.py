## needed imports
import numpy as np
import scipy as sp
import seemps as cmps
import os

## nice to haves
import matplotlib.pyplot as plt
import numba as nb
from Live_Wire.LWL import *

## sometimes used and didnt want to copy everywhere
import sympy
import time
import cProfile



ΧMax = 1000
ν = 1e-25

bonds = []

class TT():
    xfs, x0s = [], []
    Train = []
    Ns = []
    order = []
    levels = []
    tol = ν
    ChiMax = ΧMax


    DER = None
    LAP = None
    H = None

    FFT = None
    iFFT = None

    def __init__(self, T, N_s, O, L, xfs, x0s, tol=ν, ChiMax=ΧMax):
        self.Train = T
        self.Ns = N_s
        self.order = O
        self.levels = L
        self.xfs = xfs
        self.x0s = x0s
        self.tol = tol
        self.ChiMax = ChiMax

    def load(self, MD_potential=lambda x:np.sum(np.array(x)**2), tol=None):
        if self.DER == None:
            self.DER = self._MD_Derive_MPO(mode='center')
        if self.LAP is None:
            self.LAP = self._MD_Laplacian()

        if self.H is None:
            self.H = self.MD_Hamiltonian(MD_potential, tol=tol)

        if self.FFT is None:
            self.FFT = self._MD_Fourier_MPO(inv=False)
        if self.iFFT is None:
            self.iFFT = self._MD_Fourier_MPO(inv=True) 

    def stretch(self, xfs_new, x0s_new):  # changes ranges and updates derivative MPO
        self.xfs = xfs_new
        self.x0s = x0s_new
        if self.DER != None:
            self.DER = self._MD_Derive_MPO(mode='center')
        self.LAP = None
        self.H = None

    def sweep(self):
        self.Train = Sweep(self.Train, tol=self.tol, ChiMax=self.ChiMax)

    def apply(self, MPO, dim=0, lis=False):
        if lis:
            for mpo in MPO:
                self.Train = Apply_MPO(mpo, self.Train, tol=self.tol, ChiMax=self.ChiMax)
        elif not(lis):
            if len(MPO) != np.sum(self.Ns):
                print('Had to convert for you stupid')
                MPO = MD_MPO(MPO, self.order, self.levels, dim)        
            self.Train = Apply_MPO(MPO, self.Train, tol=self.tol, ChiMax=self.ChiMax)

    def flip(self, dim=0):  # only flips one dimention
        self.levels = np.array([self.Ns[self.order[k]] - 1 - self.levels[k] if dim == self.order[k] else self.levels[k] for k in range(len(self.order))])
        #return Flip_Train_MD(self)
    
    def momentum(self, mode='center'):  # isnt actually momentum or anyhit like it, the operator is 3 derivatives which gives an multiplication of the momentum expectaion values maybe?
        if self.DER == None:
            self.DER = self._MD_Derive_MPO(mode=mode)

        Conj = Conjugate_Train(self.Train)
        Deriv = Apply_MPO(self.DER, self.Train, tol=self.tol, ChiMax=self.ChiMax)

        #return np.real(-1.0*self._integrate(Multiply_Trains(Conj, Deriv, tol=self.tol, ChiMax=self.ChiMax)))
        return np.real(-1.0j * self.contract(Conj, Deriv) * np.prod(self.xfs - self.x0s) / 2**len(self.Train))

    def momentum_vec(self):  # not tested and theoreticly works?
        momentum = []

        for (i, N) in enumerate(self.Ns):
            DER_xi = MD_MPO(Derive_MPO(N, self.xfs[i], self.x0s[i]), self.order, self.levels, i)
            Conj = Conjugate_Train(self.Train)
            Deriv = Apply_MPO(DER_xi, self.Train, tol=self.tol, ChiMax=self.ChiMax)

            momentum.append(-1.0j*self._integrate(Multiply_Trains(Conj, Deriv, tol=self.tol, ChiMax=self.ChiMax)))
            #print("momentum: " , i, np.real(-1.0j*self.contract(Conj, Deriv) * np.prod(self.xfs - self.x0s) / 2**sum(self.Ns)), np.real(momentum[-1]))
            #momentum.append(-1.0j*self.contract(Conj, Deriv) * np.prod(self.xfs - self.x0s) / 2**sum(self.Ns))

        return np.real(np.array(momentum))

    @staticmethod
    def _contract_step(A, B):  # taken from Combine_MPO_step and changed to fit two trains
        chi1, _, chi2 = A.shape
        eta1, _, eta2 = B.shape

        M = np.tensordot(A, B, axes=([1], [1])).transpose(0, 2, 1, 3).reshape(chi1*eta1, chi2*eta2)
        return M

    def contract(self, Train1, Train2):

        C = self._contract_step(Train1[0], Train2[0])
            
        for (A, B) in zip(Train1[1::], Train2[1::]):
            C = C @ self._contract_step(A, B)
            
        return np.trace(C)

    # Note: Chi<30 contract faster then integrate
    #       30<Chi<100 integrate faster IF compress=False for apply, othhrwise contract is
    #       Chi>100 integrate is faster

    def _integrate(self, Train=None):
        if self.FFT is None:
            self.FFT = self._MD_Fourier_MPO(inv=False)

        try:
            New_Train = Apply_MPO(self.FFT, Train, tol=self.tol, ChiMax=self.ChiMax, compress=False)
        except:
            New_Train = Apply_MPO(self.FFT[0], self.Train, tol=self.tol, ChiMax=self.ChiMax)
            for T in self.FFT[1::]:
                New_Train = Apply_MPO(T, New_Train, tol=self.tol, ChiMax=self.ChiMax)

        A = New_Train[0][:, 0, :]
        for T in New_Train[1::]:
            A = A@T[:, 0, :]

        res = np.trace(A)
        return res * np.prod(self.xfs - self.x0s) / np.sqrt(2**sum(self.Ns))

    def _MD_Derive_MPO(self, mode='center'):
        MPO = MD_MPO(Derive_MPO(self.Ns[0], self.xfs[0], self.x0s[0], mode=mode), self.order, self.levels, 0)
        for i in range(1, len(self.Ns)):
            MPO = Combine_MPO(MPO, MD_MPO(Derive_MPO(self.Ns[i], self.xfs[i], self.x0s[i], mode=mode), self.order, self.levels, i), self.tol, self.ChiMax)
        return Sweep_MPO(MPO, tol=self.tol, ChiMax=self.ChiMax)

    def _MD_Fourier_MPO(self, inv=False):
        MPOs = []
        MPOs.append(MD_MPO(Fourier_MPO(self.Ns[0], tol=self.tol, ChiMax=self.ChiMax, inv=inv), self.order, self.levels, 0))
        for i in range(1, len(self.Ns)):
            MPOs.append(MD_MPO(Fourier_MPO(self.Ns[i], tol=self.tol, ChiMax=self.ChiMax, inv=inv), self.order, self.levels, i))
        return MPOs

    def _MD_Laplacian(self):
        return Laplacian_MPO(self.Ns, self.xfs, self.x0s, self.order, self.levels)

    def MD_Hamiltonian(self, MD_potential, tol=None):
        if self.LAP == None:
            self.LAP = self._MD_Laplacian()

        if tol != None:
            return Add_MPO(Multiply_Const_MPO(self.LAP, -0.5), Multiply_MPO(Func_to_Train_MD(MD_potential, self.Ns, self.xfs, self.x0s, self.order, tol, self.ChiMax).Train, tol=tol, ChiMax=self.ChiMax), tol=tol, ChiMax=self.ChiMax)
        
        return Add_MPO(Multiply_Const_MPO(self.LAP, -0.5), Multiply_MPO(Func_to_Train_MD(MD_potential, self.Ns, self.xfs, self.x0s, self.order, self.tol, self.ChiMax).Train, tol=self.tol, ChiMax=self.ChiMax), tol=self.tol, ChiMax=self.ChiMax)

    def norm(self):
        #MPO1 = MPS_to_MPO(Conjugate_Train(self.Train))  # TECHNICLLY unneeded shapping, but its because i want to use MPO combining
        #MPO2 = MPS_to_MPO(self.Train, mode='left')
    
        #C = Combine_MPO_step(MPO1[0], MPO2[0])[:, 0, 0, :]
    
        #for (A, B) in zip(MPO1[1::], MPO2[1::]):
        #    C = C @ Combine_MPO_step(A, B)[:, 0, 0, :]
    
        #res = np.trace(C)
        return np.abs(self.contract(Conjugate_Train(self.Train), self.Train) * np.prod(self.xfs - self.x0s) / 2**sum(self.Ns))

    def normalize(self):
        self.Train = Multiply_Const(self.Train, 1/self.norm())

    def multiply(self, other):
        try:
            self.Train = Multiply_Trains(self.Train, other.Train, tol=self.tol, ChiMax=self.ChiMax)
        except Exception as e:
            print(f"error: {e}")
            self.Train = Multiply_Trains(self.Train, other, tol=self.tol, ChiMax=self.ChiMax)

    def energy(self, Ham=None, MD_potential=None):
        if self.H == None:
            if Ham == None:
                if MD_potential == None:
                    print('Really man?? Nothing??')
                    return 0
                else:
                    self.H = self.MD_Hamiltonian(MD_potential=MD_potential)
            else:
                self.H = Ham

        #return np.real(self._integrate(Multiply_Trains(Conjugate_Train(self.Train), Apply_MPO(self.H, self.Train, tol=self.tol, ChiMax=self.ChiMax), tol=self.tol, ChiMax=self.ChiMax)))
        return np.real(self.contract(Conjugate_Train(self.Train), Apply_MPO(self.H, self.Train, tol=self.tol, ChiMax=self.ChiMax)))






def _dedup_points_to_indices(points):
    if points.ndim == 1:
        points = points.reshape(1, -1)
    sites = points.shape[1]
    I_l = [np.unique(points[:, :k], axis=0) for k in range(sites)]
    I_g = [np.unique(points[:, (k + 1):sites], axis=0) for k in range(sites)]
    #print(I_l, I_g)
    return I_l, I_g

cmps.analysis.cross.cross.CrossInterpolation.points_to_indices = staticmethod(_dedup_points_to_indices)

def Corner_Pivots(Ns, permutation):
    """One MPS-site pivot per corner of the d-dim hypercube (2**d corners),
    reordered through `permutation` to match the interleaved site layout."""
    d = len(Ns)
    N_sum = int(np.sum(Ns))
    corners = np.array(np.meshgrid(*[[0, 1]] * d, indexing='ij')).reshape(d, -1).T  # (2**d, d)
    pivots = np.zeros((len(corners), N_sum), dtype=int)
    for row, c in enumerate(corners):
        unpermuted = np.concatenate([np.full(Ns[i], c[i], dtype=int) for i in range(d)])
        pivots[row] = unpermuted[permutation]
    return pivots




def Reset_Bonds():
    global bonds
    bonds.clear()
    print('Cleaned up!\n')

def Func_to_Train_cmps_Fourier_MD(f, Ns, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax, iterations=200, start=12):

    d = len(Ns)

    if xfs == None:
        xfs = [1]*d
    if x0s == None:
        x0s = [0]*d

    intervals = []

    for (i, N) in enumerate(Ns):
        intervals.append(cmps.analysis.mesh.RegularInterval(x0s[i], xfs[i], 2**N))  # equiv to np.linspace()

    N_sum = int(np.sum(Ns))

    mesh = cmps.analysis.mesh.Mesh(intervals)

    if isinstance(start, int):
        initial_N = [start//d]*d
    else:
        initial_N = start

    permutation = cmps.analysis.mesh.interleaving_permutation(initial_N)
    map_matrix = cmps.analysis.mesh.mps_to_mesh_matrix(initial_N, permutation)

    black_box = cmps.analysis.cross.BlackBoxLoadMPS(
        lambda tensor: f(*tensor), mesh, map_matrix=map_matrix, physical_dimensions=[2] * np.sum(initial_N)
    )  # creation of function order

    Strat = cmps.cython.core.Strategy(
        normalize=False,
        tolerance=tol,
        simplification_tolerance=tol,  # square them???
        max_bond_dimension=ChiMax
    )

    strategy = cmps.analysis.cross.CrossStrategyDMRG(
        strategy=Strat,
        tol=tol,
        range_iters=(1, iterations),
        range_max_bonds=(1, ChiMax),
    )  # cross multiplication method

    result = cmps.analysis.cross.cross_dmrg(black_box, cross_strategy=strategy)  # creation of MPS

    Space = cmps.analysis.space.Space(initial_N, [(x0, xf) for (xf, x0) in zip(xfs, x0s)])

    New_result = cmps.analysis.interpolation.fourier_interpolation(result.mps, Space, initial_N, Ns, strategy=Strat)

    return TT(Multiply_Const([np.asarray(m) for m in New_result._data], 1/np.sqrt(2)**(N_sum-np.sum(initial_N))), Ns, *Interleave_Order(Ns), xfs, x0s, tol=tol, ChiMax=ChiMax)

def Func_to_Train_cmps_MD(f, Ns, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax, iterations=200, initial_bits=None, blocked = False):

    d = len(Ns)

    if xfs == None:
        xfs = [1]*d
    if x0s == None:
        x0s = [0]*d

    intervals = []

    for (i, N) in enumerate(Ns):
        intervals.append(cmps.analysis.mesh.RegularInterval(x0s[i], xfs[i], 2**N))  # equiv to np.linspace()

    N_sum = int(np.sum(Ns))
    
    mesh = cmps.analysis.mesh.Mesh(intervals)
    if blocked:
        map_matrix = cmps.analysis.mesh.mps_to_mesh_matrix(Ns)
    else:
        permutation = cmps.analysis.mesh.interleaving_permutation(Ns)
        map_matrix = cmps.analysis.mesh.mps_to_mesh_matrix(Ns, permutation)


    black_box = cmps.analysis.cross.BlackBoxLoadMPS(
        lambda tensor: f(*tensor), mesh, map_matrix=map_matrix, physical_dimensions=[2] * N_sum
    )  # creation of function order

    Strat = cmps.cython.core.Strategy(
        normalize=False,
        tolerance=tol,
        simplification_tolerance=tol,
        max_bond_dimension=ChiMax
    )  # compressing method

    strategy = cmps.analysis.cross.CrossStrategyDMRG(
        strategy=Strat,
        tol=tol,
        range_iters=(1, iterations),
        range_max_bonds=(1, ChiMax),
    )  # cross multiplication method

    if(initial_bits is None):  # default picks center of graph
        #left_edge_idx = 2**(N_sum-1) - 1  # for N = 10, edge = 511
        init_pt = np.array([1]*len(Ns) + [0]*(np.sum(Ns)-len(Ns)), dtype=int)
    elif(initial_bits == 'corn'):
        if blocked:
            init_pt = Corner_Pivots(Ns, np.arange(N_sum))
        else:
            init_pt = Corner_Pivots(Ns, permutation)
    else:
        init_pt = np.asarray(initial_bits, dtype=int)

    result = cmps.analysis.cross.cross_dmrg(black_box, cross_strategy=strategy, initial_points=init_pt)

    mps = result.mps  # getting MPS

    return TT([np.asarray(mps[i]) for i in range(len(mps))], Ns, *(Blocked_Order(Ns) if blocked else Interleave_Order(Ns)), xfs, x0s, tol=tol, ChiMax=ChiMax)

def Func_to_Train_cmps_MD_prod(f, Ns, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax, debug=0):
    d = np.sum(Ns)

    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    Train = MD_Train(Func_to_Train_cmps(f, Ns[0], xf=xfs[0], x0=x0s[0], tol=tol, ChiMax=ChiMax), Ns, 0)

    for (i, N) in enumerate(Ns[1::], 1):
        Temp_T = MD_Train(Func_to_Train_cmps(f, N, xf=xfs[i], x0=x0s[i], tol=tol, ChiMax=ChiMax), Ns, i)
        Train = Multiply_Trains(Train, Temp_T, tol=tol, ChiMax=ChiMax)

    return Train

def Func_to_Train_cmps_Fourier(f, N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, iterations=200, start = 10):

    if N <= start:
        return Func_to_Train_cmps(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)

    N = int(N)

    interval = cmps.analysis.mesh.RegularInterval(x0, xf, 2**start)  # equiv to np.linspace()
    mesh = cmps.analysis.mesh.Mesh([interval])
    map_matrix = cmps.analysis.mesh.mps_to_mesh_matrix([start])
    black_box = cmps.analysis.cross.BlackBoxLoadMPS(
        lambda tensor: f(tensor[0]), mesh, map_matrix=map_matrix, physical_dimensions=[2] * start
    )  # creation of function order

    Strat = cmps.cython.core.Strategy(
        normalize=False,
        tolerance=tol,
        simplification_tolerance=tol,  # square them???
        max_bond_dimension=ChiMax
    )

    strategy = cmps.analysis.cross.CrossStrategyDMRG(
        strategy=Strat,
        tol=tol,
        range_iters=(1, iterations),
        range_max_bonds=(1, ChiMax),
    )  # cross multiplication method

    result = cmps.analysis.cross.cross_dmrg(black_box, cross_strategy=strategy)  # creation of MPS

    Space = cmps.analysis.space.Space([start], [[x0, xf]])

    New_result, New_Space = cmps.analysis.interpolation.fourier_interpolation_1D(result.mps, Space, start, N, 0, strategy=Strat)

    return Multiply_Const([np.asarray(m) for m in New_result._data], 1/np.sqrt(2)**(N-start))

def Func_to_Train_cmps(f, N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, iterations=200, initial_bits=None):

    N = int(N)

    interval = cmps.analysis.mesh.RegularInterval(x0, xf, 2**N)  # equiv to np.linspace()
    mesh = cmps.analysis.mesh.Mesh([interval])
    map_matrix = cmps.analysis.mesh.mps_to_mesh_matrix([N])
    black_box = cmps.analysis.cross.BlackBoxLoadMPS(
        lambda tensor: f(tensor[0]), mesh, map_matrix=map_matrix, physical_dimensions=[2] * N
    )  # creation of function order

    Strat = cmps.cython.core.Strategy(
        normalize=False,
        tolerance=tol,
        simplification_tolerance=tol,
        max_bond_dimension=ChiMax
    )  # compressing method

    strategy = cmps.analysis.cross.CrossStrategyDMRG(
        strategy=Strat,
        tol=tol,
        range_iters=(1, iterations),
        range_max_bonds=(1, ChiMax),
    )  # cross multiplication method


    if(initial_bits is None):  # default picks center of graph
        left_edge_idx = 2**(N-1) - 1  # for N = 10, edge = 511
        init_pt = np.array([not ((left_edge_idx >> (N-1-k)) & 1) for k in range(N)], dtype=int)
    else:
        init_pt = np.array(initial_bits, dtype=int)

    result = cmps.analysis.cross.cross_dmrg(black_box, cross_strategy=strategy, initial_points=init_pt.reshape(1, -1))

    mps = result.mps  # getting MPS

    return [np.asarray(mps[i]) for i in range(len(mps))]




# Functions
def Gaussian(x, A=None, b=0.5, σ=0.01, k0=0):  # amplitude, center, Standard deviation, phase
    if k0 == 0:
        if A == None:
            return (1/((np.pi*σ**2)**0.25)) * np.exp(-(x-b)**2 / (2*σ**2))
        else:
            return A*np.exp(-(x-b)**2 / (2*σ**2))
    else:
        if A == None:
            return (1/((np.pi*σ**2)**0.25)) * np.exp(-(x-b)**2 / (2*σ**2)) * np.exp(1j*k0*x)
        else:
            return A*np.exp(-(x-b)**2 / (2*σ**2)) * np.exp(1j*k0*x)

def f1(x):
    return np.cos(2*np.pi*x) + np.exp(-(x-0.465)**2/0.025**2)

def f2(x):
    return np.cos(2.3434*np.pi*x**2) + 0.1*np.cos(20.3434*np.pi*x)

def f3(x):
    return -np.arctan(2.3434*np.pi*x**2) - np.cos(4.7*np.pi*x**4) - np.exp(-1.7*np.pi*(x-0.3)**2 / (0.0011)) - np.exp(-5.4*np.pi*(x-0.896)**2 / (0.001))

def f4(x):
    return np.exp(1.34*x) * (np.sin(2.1754*np.pi*x**2.1)/2 + np.cos(19.6*np.pi*x**1.9)/5)#/((x+0.1)**0.3)  Taken out so that df4 would be easier to analyticlly find

def f5(x):
    return np.exp(-(x-0.9)**2 / (0.0001)) + np.tan(x**2) * 2*np.cos(5.123*((np.pi/2)*((x+0.65))**-2)**2) - np.exp(-1.1*x) - np.exp(-3.23*(x-0.32)**2 / (0.0005))

def der_f4(x):
    A = 1.34*np.exp(1.34*x)*(0.5*np.sin(2.1754*np.pi*(x**2.1)) + 0.2*np.cos(19.6*np.pi*(x**1.9)))
    B = np.exp(1.34*x) * (1.05*2.1754*np.pi*(x**1.1)*np.cos(2.1754*np.pi*(x**2.1)) - 1.9*0.2*19.6*np.pi*(x**0.9)*np.sin(19.6*np.pi*(x**1.9)))
    return A + B

def der_der_f1(x):
    return -39.47841*np.cos(2*np.pi*x) - 3200*(-3200*np.exp(-(x-0.465)**2/0.025**2)*x**2 + 2976*np.exp(-(x-0.465)**2/0.025**2)*x - 690.92*np.exp(-(x-0.465)**2/0.025**2))




# Premade Train functions
def Exp_Train(N, α=1, xf=1, x0=0, shift=0):
    Train = []
    for i in range(N):
        delta = α * (xf - x0) / 2**(i+1)
        if i == 0:
            Train.append(np.array([[[np.exp(α*(x0-shift))], [np.exp(α*(x0-shift) + delta)]]]))
        else:
            Train.append(np.array([[[1.0], [np.exp(delta)]]]))
    return Train

def Sin_Train(N, α=1, φ=0, xf=1, x0=0):
    Train = []
    Train.append(np.array([[[np.sin(φ), np.cos(φ)], [np.sin(α/2 + φ), np.cos(α/2 + φ)]]]))
    for i in range(1, N-1):
        Train.append(np.array([[[1, 0],
                                [np.cos(α/2**(i+1)), -np.sin(α/2**(i+1))]],

                               [[0, 1],
                                [np.sin(α/2**(i+1)), np.cos(α/2**(i+1))]]]))

    Train.append(np.array([[[1], [np.cos(α/2**N)]], [[0], [np.sin(α/2**N)]]]))
    return Train

def Cos_Train(N, α=1, φ=0):
    return Sin_Train(N, α=α, φ=φ + np.pi/2)

def Delta_(N):
    Train = []
    for i in range(N):
        Train.append(np.array([[[1.0], [0.0]]]))
    return Train

def One_(N):
    Train = []
    for i in range(N):
        Train.append(np.array([[[1.0], [1.0]]]))
    return Train

def Rect(N, bits):
    Train = []
    for i in bits:
        Train.append(np.array([[[1.0-i], [i]]]))
    for _ in range(N-len(bits)):
        Train.append(np.array([[[1.0], [1.0]]]))
    return Train

def Invert(N, bits):
    return Sub_Trains(One_(N), Rect(N, bits))

def Zero(N, ChiMax=ΧMax):
    return [np.array([[[0.0], [0.0]]]) for _ in range(N)]

def Identity_2x2x2x2():
    return np.array([[[[1, 0], [0, 0]],
                      [[0, 0], [0, 0]]],
                      [[[0, 0], [0, 0]],
                       [[0, 0], [0, 1]]]])

def Identity_2x2x2x1():  # (2x2x2x1)
    return np.array([[[[1], [0]],
                        [[0], [0]]],
                        [[[0], [0]],
                        [[0], [1]]]])

def Gaussian_Train(N, xf=1, x0=0, A=1, b=0, σ=1, k0=0, R=40, debug=False):

    #All_waves = {}

    R = R if R%2==1 else R+1  # R should be odd to include the 0 point in x

    ks = np.linspace(-2 * np.pi, 2 * np.pi, R)

    C = Gaussian(ks-k0, A=A, b=0, σ=1/σ, k0=-b)

    #print(len(ks), len(C))

    Train = Multiply_Const(Exp_Train(N, 1j * ks[0], xf=xf, x0=x0), C[0])

    for (i, (k, c)) in enumerate(zip(ks[1::], C[1::])):
        #print(i)
        #All_waves[k] = Multiply_Const(Exp_Train(N, 1j * k, xf=xf, x0=x0), c)
        Train = Add_Trains(Train, Multiply_Const(Exp_Train(N, 1j * k, xf=xf, x0=x0), c), debug=debug)
        #print(Max_Bond(Train))

    #Train = Multiply_Const(Train, np.sqrt(σ))  # fixed amplitude after sigma changes
    #Train = Multiply_Const(Train, np.sqrt(2*σ/np.pi))  # fixed norm for fourier
    #Train = Multiply_Const(Train, 2*np.pi/(R))  # normal fix for fourier transform
    #return Train

    return Multiply_Const(Train, σ*2*np.sqrt(2*np.pi)/(R-1))#, All_waves

def Quickie(Train):
    X, Y = Train_to_Func(Train)
    plt.plot(X, np.real(Y))
    plt.show()

def Quickie2(Train, func=np.real):
    XY, z = Train_to_Func_MD(Train)
    Graph2(XY, func(z))

def Graph2(XY, z, log=False):
    x, y = np.real(XY[:, 0]), np.real(XY[:, 1])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=40, depthshade=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if(log):
        ax.set_zscale('log')
        ax.set_zlim(1e-20)
    ax.set_title('3D Scatter of (x, y) Coordinates vs z Value')

    fig.colorbar(scatter, ax=ax, shrink=0.6, label='z value')

    plt.tight_layout()
    plt.show()




# Orders and Levels
def Blocked_Order(Ns):
    order = np.concatenate([[dim]*Ns[dim] for dim in range(len(Ns))])
    levels = np.concatenate([np.arange(Ns[dim]) for dim in range(len(Ns))])
    return order, levels

def Interleave_Order(Ns):  # Ns: list[int], per-dimension bit depths
    order, levels = [], []
    for level in range(max(Ns)):
        for dim in range(len(Ns)):
            if level < Ns[dim]:
                order.append(dim)
                levels.append(level)
    return np.array(order), np.array(levels)

def Levels_from_order(order):
    order = np.asarray(order)
    levels = np.zeros(len(order), dtype=int)
    for dim in np.unique(order):
        mask = order == dim
        levels[mask] = np.arange(np.sum(mask))
    return levels

def Starting_vals(Ns, xfs, x0s):
    d = len(Ns)

    try:  # if they are None, set defaults
        if xfs == None:
            xfs = [1]*d
    
        if x0s == None:
            x0s = [0]*d
    except:  # if they are inputted as numpy array and python gets cranky cuz its not a bool type (I know this should be simplifyable but its been 15 minutes and i want this to run)
        if any(xfs == None):
            xfs = [1]*d

        if any(x0s == None):
            x0s = [0]*d

    return np.array(xfs), np.array(x0s)

    





# Set-up/Set-down Train
def Func_to_Train_Custum_Pot(dt, N, xf=1, x_cutoff=1, tol=ν, ChiMax=ΧMax, start=10, debug=False, Mine = True):  # fix with rect functions, faster then everything else before

    x0 = -xf
    dx = (xf-x0)/(2**(N))

    def F(k_):  # linear (abs)
        return np.where(np.abs(k_) < x_cutoff, k_**2, -x_cutoff*(x_cutoff-2*np.abs(k_)))

    def f(k):
        return np.exp(-0.5j*F(k)*dt)

    if N <= start:
        return Func_to_Train(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
    else:
        Val = N

        for i in range(N)[::-1]:
            if xf / (2**(N-i)) < x_cutoff:  # if N=11 xf = 50, and x_cutoff wanted is 20, then break if i==10 so tempx = 25
                Val = i + 1
                break

        if Val >= N:  # backup incase its just better to do whole function anyways
            print('Skipped')
            if Mine == True:
                return  Func_to_Train_Fourier(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, start=start)
            else:
                return  Func_to_Train_cmps(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
        elif Val < start:  # if it wants N=6 from -5 to 5, just do N=10 from -100 to 100. Stops small errors close to x_cutoff
            Val = start

        tempx = xf / (2**(N-Val))
        if Mine == True:
            Train = Func_to_Train_Fourier(f, Val, xf=tempx, x0=-tempx, tol=tol, ChiMax=ChiMax, start=start)  # get good from the cutoff points
        else:
            Train = Func_to_Train_cmps(f, Val, xf=tempx, x0=-tempx, tol=tol, ChiMax=ChiMax)  # get good from the cutoff points

        print('Cut off Value: ', Val, '; Temporary X: ', tempx)

        if(debug):
            print('Half way bond before: ', Max_Bond(Train))

        Train = Sweep(Train, tol=tol, ChiMax=ChiMax)

        if(debug):
            print('Half way bond after: ', Max_Bond(Train))

        # Prepping Original center of train
        Train = Apply_MPO(Twos_comp_MPO(Val), Train, tol=tol, ChiMax=ChiMax, compress=False)
        bond = Train[0].shape[-1]
        T = np.zeros((bond, 2, bond), dtype=complex)
        T[:, 1, :] = np.identity(bond, dtype=complex)
        for i in range(Val, N):
            Train.insert(1, T)  # _F_F'

        TWO = Twos_comp_MPO(N)

        Train = Apply_MPO(TWO, Train, tol=tol, ChiMax=ChiMax, compress=False)

        a = -x_cutoff*dt*1.0j
        shift = x_cutoff/2

        Right = Exp_Train(N-1, α=a, xf=xf, x0=0, shift=shift)  # bond = 2
        Left = Exp_Train(N-1, α=a, xf=xf+dx, x0=dx, shift=shift)  # bond = 2

        T = np.array([[[0.0+0.0j], [1.0+0.0j]]], dtype=complex)

        Right.insert(0, T)
        Left.insert(0, T)
        Left = Apply_MPO(TWO, Left, tol=tol, ChiMax=ChiMax, compress=False)  # bond = 2*2
        Left = Chop_Train(Left, 0)

        Train_temp = Add_Trains(Right, Left, tol=tol, ChiMax=ChiMax, compress=False)  # bond = 2**4 = 16, dont compress this yet

        bits = np.array([0] + [1 for _ in range(Val, N)])
        Square = Sub_Trains(One_(N), Add_Trains(Rect(N, bits), Rect(N, 1-bits)))

        Train_temp = Multiply_Trains(Train_temp, Square)

        Wave = Add_Trains(Train, Train_temp, tol=tol, ChiMax=ChiMax)

        return Wave

def Func_to_Train_Custum(f, N, xf=1, x_cutoff=1, tol=ν, ChiMax=ΧMax, start=10, debug=0):
    x0 = -xf
    if N < start+1:
        return Func_to_Train(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
    else:
        Val = N

        for i in range(N)[::-1]:
            if xf / (2**(N-i)) < x_cutoff:  # if N=11 xf = 50, and x_cutoff wanted is 20, then break if i==10 so tempx = 25
                Val = i + 1
                break

        if Val >= N:
            return  Func_to_Train_Fourier(f, N, xf=xf, x0=x0, tol=tol)
        if Val < start:  # if it wants N=6 from -5 to 5, just do N=10 from -100 to 100. Stops small errors close to x_cutoff
            Val = start

        tempx = xf / (2**(N-Val))
        Train = Func_to_Train_Fourier(f, Val, xf=tempx, x0=-tempx, tol=tol)

        print('Cut off Value: ', Val, '; Temporary X: ', tempx)

        Train = Apply_MPO(Twos_comp_MPO(Val), Train)

        bond = Train[1].shape[0]
        T = np.zeros((bond, 2, bond), dtype=complex)
        T[:, 1, :] = np.identity(bond, dtype=complex)
        for i in range(Val, N):
            Train.insert(1, T)

        Train = Apply_MPO(Twos_comp_MPO(N), Train)

        return Sweep(Train, tol=tol, ChiMax=ChiMax)

def Func_to_Train_Zero_fill(f, N, xf=1, tol=ν, ChiMax=ΧMax, start=10, debug=0):  # only works for symetric functions as of now** DO NOT USE, not stable
    x0 = -xf
    if N < start+1:
        return Func_to_Train(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
    else:
        Train = Func_to_Train(f, start, xf=xf/(2**(N-start)), x0=x0/(2**(N-start)), tol=tol, ChiMax=ChiMax, debug=debug)
        Train = Apply_MPO(Twos_comp_MPO(start), Train)

        bond = Train[1].shape[0]
        T = np.zeros((bond, 2, bond), dtype=complex)
        T[:, 1, :] = np.identity(bond, dtype=complex)
        for i in range(start, N):
            Train.insert(1, T)

        Train = Apply_MPO(Twos_comp_MPO(N), Train)

        return Train

def Func_to_Train_Fourier(f, N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, start=10, debug=0):  #
    if N < start+1:
        return Func_to_Train(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
    else:
        Train = Func_to_Train(f, start, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
        Train = Apply_MPO(Fourier_MPO(start, tol=tol, ChiMax=ChiMax), Train, tol=tol, ChiMax=ChiMax, debug=debug)
        Train = Apply_MPO(Flip_MPO(Twos_comp_MPO(start)), Train)

        bond = Train[-1].shape[0]
        T = np.zeros((bond, 2, bond), dtype=complex)
        T[:, 0, :] = np.identity(bond, dtype=complex)
        for i in range(start, N):
            Train.insert(i-1, T)

        Train = Apply_MPO(Flip_MPO(Twos_comp_MPO(N)), Train)
        Train = Multiply_Const(Apply_MPO(Fourier_MPO(N, inv=True, tol=tol, ChiMax=ChiMax), Train), np.sqrt(2)**(N-start))
        return Train

def Func_to_Train_Interpolation(f, N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, start=10, debug=0):  # stops calling MPO every time but has higher bond dimention (3)

    if N < start+1:
        return Func_to_Train(f, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
    else:
        Train = Func_to_Train(f, start, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, debug=debug)
        MPO = Shift_minus_plus_Iden_MPO(start-1)
        MPO = Multiply_Const_MPO(MPO, 0.5)
        for i in range(start+1, N+1):

            MPO.insert(i-3, Shift_minus_plus_Iden_MPO_step())
            Train0 = Train + [np.array([[[1], [0]]])]
            temp = Apply_MPO(MPO, Train, tol=tol, ChiMax=ChiMax, debug=debug) + [np.array([[[0], [1]]])]
            Train = Add_Trains(Train0, temp, tol=tol, ChiMax=ChiMax, debug=debug)

        return Train

def One_rip(F, N, tol, Chilast, ChiMax):  # Rip one matrix off from F, Step N
    U, σ, V = np.linalg.svd(F.reshape((2*Chilast, -1)), full_matrices=False)

    err = np.append(np.cumsum(σ[::-1]**2)[::-1], 0)
    max_err = tol*err[0]
    Chi_cut = np.argmax(err<max_err)
    Chi = max(min(ChiMax, len(σ), Chi_cut), 1)

    return np.reshape(U[:, :Chi], (Chilast, 2, Chi)), σ[:Chi, None] * V[:Chi, :], Chi  # changed to allow higher N to be used (use less ram by not asking for a (2^N, 2^N matrix every time))

def Func_to_Train(f, N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, compress=True, debug=0):  # Rip up into a matrix train
    k = np.arange(2**N)
    bits = ((k[:, None] >> np.arange(N)) & 1)          # shape (2^N, N), LSB first
    xt = (xf - x0) * (bits * 0.5**np.arange(1, N+1)).sum(axis=1) + x0

    F = f(xt).reshape([2]*N, order='F')

    Temp, F, last = One_rip(F, N, tol, 1, ChiMax)  # ADDED week 4

    train = [Temp]  # changed week 4

    for i in range(1, N-1):
        Temp, F, last = One_rip(F, N-i, tol, last, ChiMax)
        train.append(Temp)
        Compression_step(train, i-1, tol=tol, ChiMax=ChiMax, debug=debug, mode='right')  # ADDED week 4

    train.append(F.reshape((*(np.shape(F)), 1)))
    Compression_step(train, N-2, tol=tol, ChiMax=ChiMax, debug=debug, mode='right')

    if (debug == 1):
        for Temp in train:
            print(Temp.shape)
        print('')

    if not compress:
        return train
    else:
        return Sweep(train, tol=tol, ChiMax=ChiMax)

def Train_to_Func(Train, xf=1, x0=0, Flip=0, debug=False, N_res=18):
    if(Flip == 1):
        Train = Flip_Train(Train)
    
    N = len(Train)
    Xi,Yi = [], []

    if (N > N_res):

        A0 = Train[-1][:, 0, :]

        for i in range(N_res, N-1)[::-1]:
            A0 = Train[i][:, 0, :] @ A0

        T = np.zeros((Train[N_res-1].shape[0], 2, 1), dtype=complex)

        T[:, 0, :] = Train[N_res-1][:, 0, :] @ A0
        T[:, 1, :] = Train[N_res-1][:, 1, :] @ A0

        Train = Train[:N_res-1:] + [T]

        N = N_res


    for i in range(2**N):

        bit_str = (bin(i)[2:].zfill(N))[::-1]
        index = tuple([int(b) for b in bit_str])
        x = np.sum([(int(b)*(xf-x0))/2**(n+1) for (n, b) in enumerate(bit_str)]) + x0
        Xi.append(x)  # [np.float64(0.0), np.float64(0.5), np.float64(0.25), np.float64(0.75), ...]


        Ans_T = Train[0][:, index[0], :]
        for j in range(1, len(Train)-1):
            Ans_T = Ans_T @ Train[j][:, index[j], :]


        Ans_T = Ans_T @ Train[-1][:, index[-1], :]

        Yi.append(np.trace(Ans_T))

    if(debug):
        print(len(Xi), len(Yi))

    x0, y0 = Xi,Yi

    pairs = list(zip(np.array(Xi),np.array(y0)))
    pairs = sorted(pairs, key=lambda x: x[0])
    Xf = np.array([np.real(p[0]) for p in pairs])
    Yf = np.array([p[1] for p in pairs])

    return Xf,Yf


def Func_to_Train_MD(f, Ns, xfs=None, x0s=None, order=None, tol=ν, ChiMax=ΧMax, compress=True, debug=False):
    if order is None:
        order, levels = Interleave_Order(Ns)
    elif callable(order):
        order, levels = order(Ns)
    elif isinstance(order, Iterable):
        levels = Levels_from_order(order)
    else:
        print('no levels\n')
        return False

    
    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    N_sum = int(np.sum(Ns))
    
    k = np.arange(2**N_sum)
    bits = ((k[:, None] >> np.arange(N_sum)) & 1)

    weights = 0.5 ** (levels + 1)  # value of any given bit

    xt_lists = []
    
    for i in range(len(Ns)):
        mask = (order == i)
        xt = (xfs[i] - x0s[i]) * (bits[:, mask] * weights[mask]).sum(axis=1) + x0s[i]
        xt_lists.append(xt)
    
    F = f(*xt_lists).reshape([2]*N_sum, order='F')

    Temp, F, last = One_rip(F, N_sum, tol, 1, ChiMax)
    train = [Temp]

    for i in range(1, N_sum-1):
        Temp, F, last = One_rip(F, N_sum-i, tol, last, ChiMax)
        train.append(Temp)
        Compression_step(train, i-1, tol=tol, ChiMax=ChiMax, debug=debug, mode='right')

    train.append(F.reshape((*np.shape(F), 1)))
    Compression_step(train, N_sum-2, tol=tol, ChiMax=ChiMax, debug=debug, mode='right')

    if not compress:
        return TT(train, Ns, order, levels, xfs, x0s)
    else:
        return TT(Sweep(train, tol=tol, ChiMax=ChiMax), Ns, order, levels, xfs, x0s)

def Train_to_Func_MD(TT, xfs=None, x0s=None, Flip=None, debug=False, N_res=18):
    Train = copyT(TT.Train)
    Ns = TT.Ns
    order = TT.order
    levels = TT.levels
    N = len(Train)

    if Flip is None:
        Flip = np.zeros(len(Ns), dtype=int)
    else:
        Flip = np.asarray(Flip)

    levels = np.array([
        Ns[order[k]] - 1 - levels[k] if Flip[order[k]] else levels[k]
        for k in range(len(order))
    ])

    Ns = list(Ns)
    order = list(order)
    levels = list(levels)


    while(sum(Ns) > N_res):
        

        remed = levels.index(int(np.max(Ns)-1))  # site being removed

        Rem = Train.pop(remed)[:, 0, :]

        if remed != 0:  # combine with left one as default
            T = np.zeros((Train[remed-1].shape[0], 2, Rem.shape[-1]), dtype=complex)

            T[:, 0, :] = Train[remed-1][:, 0, :] @ Rem
            T[:, 1, :] = Train[remed-1][:, 1, :] @ Rem

            Train[remed-1] = T

        else:  # combine with right side instead
            T = np.zeros((1, 2, Train[0].shape[-1]), dtype=complex)  # rem.shape[0] should be 1 so no problems??
            
            T[:, 0, :] = Rem @ Train[remed][:, 0, :]
            T[:, 1, :] = Rem @ Train[remed][:, 1, :]

            Train[remed] = T

        

        

        Ns[order[remed]] -= 1

        order.pop(remed)
        levels.pop(remed)

    Ns = np.array(Ns)
    order = np.array(order)
    levels = np.array(levels)
    N = len(Train)

    
    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    

    coords = []
    vals = []

    for i in range(2**N):
        bit_str = (bin(i)[2:].zfill(N))[::-1]
        index = tuple(int(b) for b in bit_str)
        bit_arr = np.array(index)

        Ans_T = Train[0][:, index[0], :]
        for j in range(1, len(Train)-1):
            Ans_T = Ans_T @ Train[j][:, index[j], :]
        Ans_T = Ans_T @ Train[-1][:, index[-1], :]
        val = np.trace(Ans_T)

        cord = np.array([0.0]*len(Ns))  # single corrdinate for this set of tensors
        for (k, which_cod) in enumerate(order):
            cord[which_cod] += int(index[k]) / (2**(levels[k]+1))

        cord *= (xfs-x0s)
        cord += x0s

        coords.append(tuple(cord))
        vals.append(val)

    #if Flip == 1:
    #    coords = [c[::-1] for c in coords]

    return np.array(coords), np.array(vals)

def Func_to_Train_Fourier_MD(f, Ns, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax, start=12):  # not setup for blocked order
    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    if np.sum(Ns) < start:
        return Func_to_Train_MD(f, Ns, xfs=xfs, x0s=x0s, tol=tol, ChiMax=ChiMax)
    else:
        d = len(Ns)
        start_Ns = [min(N, start // d) for N in Ns]
        Train = Func_to_Train_MD(f, start_Ns, xfs=xfs, x0s=x0s, tol=tol, ChiMax=ChiMax)


        Train.Train = Apply_MPO(MD_Fourier_MPO(start_Ns, Train.order, Train.levels, tol=tol, ChiMax=ChiMax),
                             Train.Train, tol=tol, ChiMax=ChiMax)
        Train.Train = Apply_MPO(MD_TwosComp_MPO(start_Ns, Train.order, Train.levels, tol=tol, ChiMax=ChiMax),
                             Train.Train, tol=tol, ChiMax=ChiMax)

        pads = []
        for dim in range(d):
            if Ns[dim] > start_Ns[dim]:
                pos = int(np.where((Train.order == dim) & (Train.levels == start_Ns[dim] - 1))[0][0])
                pads.append((pos, Ns[dim] - start_Ns[dim]))

        for (pos, extra) in sorted(pads, reverse=True):  # highest position first so lower positions stay valid
            bond = Train.Train[pos].shape[0]
            T = np.zeros((bond, 2, bond), dtype=complex)
            T[:, 0, :] = np.identity(bond, dtype=complex)
            for _ in range(extra):
                Train.Train.insert(pos, T)

        Train.Ns = list(Ns)
        Train.order, Train.levels = Interleave_Order(Ns)

        Train.Train = Apply_MPO(MD_TwosComp_MPO(Ns, Train.order, Train.levels, tol=tol, ChiMax=ChiMax),
                                Train.Train, tol=tol, ChiMax=ChiMax)
        Train.Train = Multiply_Const(
            Apply_MPO(MD_Fourier_MPO(Ns, Train.order, Train.levels, inv=True, tol=tol, ChiMax=ChiMax),
                    Train.Train, tol=tol, ChiMax=ChiMax),
            np.sqrt(2) ** (np.sum(Ns) - np.sum(start_Ns))
        )

        Train.sweep()

        return Train

def Identity_step(chi):
    return np.eye(chi, dtype=complex).reshape(chi, 1, chi) * np.ones((1, 2, 1), dtype=complex)

def MD_Train(Train_, Ns, dim, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax):

    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    Train = []
    order, levels = Interleave_Order(Ns)
    temp = 1

    for (i, cord) in enumerate(order):
        if dim == cord:
            Train.append(Train_[levels[i]])
            temp = Train_[levels[i]].shape[-1]
        else:
            Train.append(Identity_step(temp))
    return TT(Train, Ns, order, levels, xfs, x0s, tol, ChiMax)

def Func_to_Train_MD_prod(f, Ns, xfs=None, x0s=None, tol=ν, ChiMax=ΧMax, debug=0):

    xfs, x0s = Starting_vals(Ns, xfs, x0s)

    Train = MD_Train(Func_to_Train(f, Ns[0], xf=xfs[0], x0=x0s[0], tol=tol, ChiMax=ChiMax), Ns, 0)

    for (i, N) in enumerate(Ns[1::], 1):
        Temp_T = MD_Train(Func_to_Train(f, N, xf=xfs[i], x0=x0s[i], tol=tol, ChiMax=ChiMax), Ns, i)
        Train.Train = Multiply_Trains(Train.Train, Temp_T.Train, tol=tol, ChiMax=ChiMax)

    return Train

def Func_to_Train_MD_pot(dt, Ns, xfs=None, x_cutoff=1, tol=ν, ChiMax=ΧMax, start=12, debug=0):  # not done, just a copy of fourier right now

    def f(*ks):  # kinetic
        k2 = sum(k**2 for k in ks)
        window = 1
        for k in ks:
            window = window * sp.special.expit(k + x_cutoff) * sp.special.expit(x_cutoff - k)
        return np.exp(-0.5j*k2*dt) * window


    if xfs == None:
        xfs = np.array([1]*len(Ns))
    else:
        xfs = np.asarray(xfs)

    x0s = -xfs

    if np.sum(Ns) < start+1:
        return Func_to_Train_MD(f, Ns, xfs=xfs, x0s=x0s, tol=tol, ChiMax=ChiMax, debug=debug)
    else:
        Val = np.array(Ns)

        for (n, N) in enumerate(Ns):
            for i in range(N)[::-1]:
                if xfs[n] / (2**(N-i)) < x_cutoff:  # if N=11 xf = 50, and x_cutoff wanted is 20, then break if i==10 so tempx = 25
                    Val[n] = i + 1
                    break

        if np.sum(Val) >= np.sum(Ns):
            return  Func_to_Train_Fourier_MD(f, Ns, xfs=xfs, x0s=x0s, tol=tol)
        if np.any(Val < start//len(Ns)):  # if it wants N=6 from -5 to 5, just do N=10 from -100 to 100. Stops small errors close to x_cutoff
            Val = np.maximum(Val, start//len(Ns))

        tempx = xfs / (2**(Ns-Val))
        Train = Func_to_Train_Fourier_MD(f, Val, xfs=tempx, x0s=-tempx, tol=tol)

        print('Cut off Value: ', Val, '; Temporary X: ', tempx)

        Train.Train = Apply_MPO(MD_TwosComp_MPO(Val, Train.order, Train.levels, tol=tol), Train.Train)

        pads = []
        for dim in range(len(Ns)):
            if Ns[dim] > Val[dim]:
                pos = int(np.where((Train.order == dim) & (Train.levels == Val[dim] - 1))[0][0])
                pads.append((pos, Ns[dim] - Val[dim]))

        for (pos, extra) in sorted(pads, reverse=True):
            bond = Train.Train[pos].shape[0]
            T = np.zeros((bond, 2, bond), dtype=complex)
            T[:, 1, :] = np.identity(bond, dtype=complex)
            for _ in range(extra):
                Train.Train.insert(pos, T)

        Train.order, Train.levels = Interleave_Order(Ns)
        Train.Train = Apply_MPO(MD_TwosComp_MPO(Ns, Train.order, Train.levels, tol=tol), Train.Train)

        return TT(Sweep(Train.Train, tol=tol, ChiMax=ChiMax), Ns, Train.order, Train.levels, Train.xfs, Train.x0s, tol, ChiMax)




# Help Functions
def Train_Storage(Train, vag=0):  # vag=N for possible train
    if(vag == 0):
        Stored = 0
        for T in Train:
            temp = 1
            for s in T.shape:
                temp *= s
            Stored += temp
        print(Stored)
        return Stored
    else:
        nums = 0
        for n in range(vag):
            if n < vag // 2:
                shape = [2**n, 2, 2**(n+1)]
            elif (int(n) == vag//2 and vag % 2 == 1):
                shape = [2**(vag//2), 2, 2**(vag//2)]
            else:
                shape = [2**(vag-n), 2, 2**(vag-n-1)]
            temp = 1

            for s in shape:
                temp *= s
            nums += temp
            #print(shape)
        #print("\n")
        return nums * 16 / 1000000000

def Max_Bond(Train):
    max = 0
    for T in Train:
        sh = np.shape(T)
        if sh[0] > max:
            max = sh[0]
        #if sh[-1] > max:  # ignore last bond since if its not one, i cant reduce it yet
        #    max = sh[-1]
    return max

def Train_Shape(Train, o=True):
    if(o):
        for T in Train:
            print(np.shape(T))
        print('\n')
    else:
        for T in Train:
            print(np.shape(T)[0], np.shape(T)[-1])
        print('\n')

def copyT(Train):
    return [np.copy(T) for T in Train]

def copyTT(Train):
    return TT(Train.Train, Train.Ns, Train.order, Train.levels, Train.xfs, Train.x0s, Train.tol, Train.ChiMax)

def Flip_Train(Train):
    return [T.transpose(2, 1, 0) for T in Train][::-1]

def Flip_MPO(MPO):

    New_MPO = []
    for M in MPO:
        New_MPO.append(M.transpose(3, 1, 2, 0))

    return New_MPO[::-1]

def Chop_Train(Train, Chop_site):
    new_train = copyT(Train)

    temp = new_train[Chop_site]

    tempA = temp[:, 0, :]
    tempB = temp[:, 1, :]

    α, β = tempA.shape

    temp = np.zeros((α, 2, β), dtype=complex)
    temp[:, 0, :] = tempB
    temp[:, 1, :] = tempA

    new_train[Chop_site] = temp

    return new_train

def Find_Ground_State(potential, xf, x0, N, Pos_only=True, Normalize=False, state=1, find=None):  # og=true is faster
    
    if find==None:
        find=state

    n = 2**N

    dx = (xf-x0)/n

    x_points = np.linspace(x0, xf-dx, n)

    pots = potential(x_points)

    if N < 12:
        a = 1/(12*dx**2)

        main = 30*a + pots          # length n, offset 0
        off  = -8*a * np.ones(n-1)            # length n-1, offsets +1/-1
        off2  = 0.5*a * np.ones(n-2)            # length n-1, offsets +1/-1

        H = sp.sparse.diags(
            [main, off, off, off2, off2, [-8*a], [-8*a], [0.5*a, 0.5*a], [0.5*a, 0.5*a]],
            [0,    1,   -1,   2,   -2,    n-1,   -(n-1),   n-2,   -(n-2)]
        ).tocsr()
    else:
        a = 1/dx**2

        main = a + pots          # length n, offset 0
        off  = -0.5*a * np.ones(n-1)            # length n-1, offsets +1/-1

        H = sp.sparse.diags(
            [main, off, off, [-0.5*a], [-0.5*a]],
            [0,    1,   -1,   n-1,      -(n-1)]
        ).tocsr()

    eigenvalue, eigenvector = sp.sparse.linalg.eigsh(H, k=find, which='LM', sigma=np.min(pots))

    if (Pos_only and (eigenvector[:, state-1][n//2] < 0)):  # makes sure its the positive version, just because...
        print('Flipped function')
        eigenvector = -1.0*eigenvector

    print(eigenvalue)

    if Normalize:
        return sp.interpolate.make_interp_spline(x_points, eigenvector[:, state-1]/np.sqrt(dx))
    else:
        return sp.interpolate.make_interp_spline(x_points, eigenvector[:, state-1])

def Flip_Train_MD(Train):
    Train.Train = Flip_Train(Train.Train)
    #Train.Ns = Train.Ns[::-1]
    Train.order = Train.order[::-1]
    #Train.levels = Train.levels[::-1]




# Train Info  (Do not return a train, return a value)
def Integrate_Train(Train, xf=1, x0=0, tol=ν, ChiMax=ΧMax, FFT=None, debug=False):  # Slow because of FFT, functions below avoid it
    N = len(Train)

    if FFT is None:
        FFT = Fourier_MPO(N, tol=tol, ChiMax=ChiMax)

    New_Train = Apply_MPO(FFT, Train, tol=tol, ChiMax=ChiMax, debug=debug)

    A = New_Train[0][:, 0, :]
    for T in New_Train[1::]:
        A = A@T[:, 0, :]

    res = np.trace(A)
    return res * ((xf-x0)/np.sqrt(2**N))  # added constant to counter the effect of the Fourier application

def Momentum(psi, xf=1, x0=0, tol=ν, ChiMax=ΧMax, DER=None, debug=False):  # Made linear
    if DER == None:
        DER = Derive_MPO_Direct(len(psi), xf=xf, x0=x0, mode='center')
    Conj = Conjugate_Train(psi)
    Deriv = Apply_MPO(DER, psi, tol=tol, ChiMax=ChiMax)

    MPO1 = MPS_to_MPO(Conj)
    MPO2 = MPS_to_MPO(Deriv, mode='left')

    M = Combine_MPO_step(MPO1[0], MPO2[0])[:, 0, 0, :]

    for (A, B) in zip(MPO1[1::], MPO2[1::]):
        M = M @ Combine_MPO_step(A, B)[:, 0, 0, :]

    return -1.0j * np.trace(M) * ((xf - x0) / 2**len(psi))

def Norm_Train(Train, xf=1, x0=0, debug=False):  # fixed to not use other functions
    MPO1 = MPS_to_MPO(Conjugate_Train(Train))
    MPO2 = MPS_to_MPO(Train, mode='left')

    C = Combine_MPO_step(MPO1[0], MPO2[0], debug=debug)[:, 0, 0, :]

    for (A, B) in zip(MPO1[1::], MPO2[1::]):
        C = C @ Combine_MPO_step(A, B, debug=debug)[:, 0, 0, :]

    res = np.trace(C)
    return res * (xf - x0) / 2**len(Train)

def Current(psi, xf, x0, q=1, tol=ν, ChiMax=ΧMax, DER=None, debug=False):  # Made linear, not tested
    return q * Momentum(psi, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax, DER=DER, debug=debug)




# Train Operations
def Add_step(A, B, pre=False, debug=False):
    chi1,chi2 = A[:,0,:].shape
    eta1,eta2 = B[:,0,:].shape
    #print('pre: ', pre)
    if pre == -1:  # if both first vector and dimention is 1
        if (chi1 == 1 and eta1 ==1):
            T = np.zeros((1,2,chi2+eta2), dtype=complex)
            if(debug):
                print(chi1, chi2, eta1, eta2)
                print(T.shape, 'Shape inserted 1\n')
            T[0,0,:] = np.hstack([A[0,0,:],B[0,0,:]])
            T[0,1,:] = np.hstack([A[0,1,:],B[0,1,:]])
            return T
    elif pre == 1:  # if both last vector and dimention is 1
        if (chi2 == 1 and eta2 == 1):
            T = np.zeros((chi1+eta1,2,1), dtype=complex)
            if(debug):
                print(chi1, chi2, eta1, eta2)
                print(T.shape, 'Shape inserted 2\n')
            T[:,0,0] = np.hstack([A[:,0,0],B[:,0,0]])
            T[:,1,0] = np.hstack([A[:,1,0],B[:,1,0]])
            return T

    T = np.zeros((chi1+eta1,2,chi2+eta2), dtype=complex)

    T[:,0,:] =  np.block([
        [A[:,0,:], np.zeros((chi1,eta2))],
        [np.zeros((eta1,chi2)), B[:,0,:]]
    ])
    T[:,1,:] = np.block([
        [A[:,1,:], np.zeros((chi1,eta2))],
        [np.zeros((eta1,chi2)), B[:,1,:]]
    ])

    if(debug):
        print(chi1, chi2, eta1, eta2)
        print(T.shape, 'Shape inserted\n')
    return T

def Add_Trains(Train1, Train2, tol=ν, ChiMax=ΧMax, debug=False, compress=True):
    newTrain = []

    if(debug):
        Train_Shape(Train1)
        Train_Shape(Train2)

    newTrain.append(Add_step(Train1[0], Train2[0], pre=-1, debug=debug))
    N = len(Train1)
    for (i, (T1,T2)) in enumerate(zip(Train1[1:],Train2[1:])):
        new = Add_step(T1, T2, pre=N-1-i, debug=debug)
        newTrain.append(new)
        if(compress):
            Compression_step(newTrain, i, tol=tol, ChiMax=ChiMax, debug=False)

    if(debug):
        print('Real Final shape\n')
        Train_Shape(newTrain)

    if(compress):
        return Sweep(newTrain, tol=tol, ChiMax=ChiMax)
    else:
        return newTrain

def Sub_Trains(Train1, Train2, tol=ν, ChiMax=ΧMax, debug=False, compress=True):
    return Add_Trains(Train1, Multiply_Const(Train2, -1), debug=debug, tol=tol, ChiMax=ChiMax, compress=compress)

def Multiply_step(A, B, debug=False):
    chi1,chi2 = A[:,0,:].shape
    eta1,eta2 = B[:,0,:].shape

    T = np.zeros((chi1*eta1,2,chi2*eta2), dtype=complex)

    T[:,0,:] = np.kron(A[:,0,:], B[:,0,:])
    T[:,1,:] = np.kron(A[:,1,:], B[:,1,:])

    if(debug):
        print(A.shape)
        print(B.shape)
        print(T.shape)
        print('\n')

    return T

def Multiply_Trains(Train1, Train2, tol=ν, ChiMax=ΧMax, debug=False):
    newTrain = [Multiply_step(Train1[0], Train2[0], debug=debug)]

    for (i, (T1, T2)) in enumerate(zip(Train1[1::], Train2[1::])):

        newTrain.append(Multiply_step(T1, T2, debug=debug))

        Compression_step(newTrain, i, tol=tol, ChiMax=ChiMax)
    Compression_sweep_left(newTrain, tol=tol, ChiMax=ChiMax)
    return Sweep(newTrain, tol=tol, ChiMax=ChiMax)

def Multiply_Const(Train, C):

    Train_new = copyT(Train)

    Train_new[0] = C * Train_new[0]

    return Train_new

def SecondDerive_Train(Train_I, xf=1, debug=False, tol=ν, ChiMax=ΧMax):
    Train = copyT(Train_I)
    Train_plus = Apply_MPO(Shift_MPO_plus(len(Train)), Train, tol=tol, ChiMax=ChiMax)
    Train_minus = Apply_MPO(Shift_MPO_minus(len(Train)), Train, tol=tol, ChiMax=ChiMax)
    return Multiply_Const(Add_Trains(Sub_Trains(Train_plus, Multiply_Const(Train, 2), debug=debug, tol=tol, ChiMax=ChiMax), Train_minus, debug=debug, tol=tol, ChiMax=ChiMax), ((2**(len(Train)-1))/xf)**2)

def Imaginary_Train(Train, tol=ν, ChiMax=ΧMax):
    Train_new = Multiply_Const(Sub_Trains(Train, Conjugate_Train(Train), tol=tol, ChiMax=ChiMax), -0.5j)
    return Train_new

def Conjugate_Train(Train):
    Train_new = []
    for T in Train:
        Train_new.append(np.conjugate(T))
    return Train_new

def Prob_Current_Train(psi, xf=1, x0=0, tol=ν, ChiMax=ΧMax, debug=False, DER=None):  # faster
    if DER is None:
        DER = Derive_MPO_Direct(len(psi), xf=xf, x0=x0, debug=debug, mode='center')
    return Imaginary_Train(Multiply_Trains(Conjugate_Train(psi), Apply_MPO(DER, psi, tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax)

def Prob_Current_Train2(psi, xf=1, x0=0, tol=ν, ChiMax=ΧMax, debug=False, DER=None):  # usually slower other one..
    if DER is None:
        DER = Derive_MPO_Direct(len(psi), xf=xf, x0=x0, debug=debug, mode='center')
    conj = Conjugate_Train(psi)
    return Multiply_Const(Sub_Trains(Multiply_Trains(psi, Apply_MPO(DER, conj, tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax), Multiply_Trains(conj, Apply_MPO(DER, psi, tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax), 0.5j)

def Contract_Trains(Train1, Train2, tol=ν, ChiMax=ΧMax):  # equivalent to multiply Trains???

    MPO1 = MPS_to_MPO(Conjugate_Train(Train1))
    MPO2 = MPS_to_MPO(Train2, mode='left')

    MPO = Combine_MPO(MPO1, MPO2, tol=tol, ChiMax=ChiMax)

    New_Train = MPO_to_MPS(MPO)

    return New_Train

def Train_Ground_state(Train, H):
    return 0




# Applying MPOs
def Apply_MPO_step(M, A, debug=False):
    chi1  = A.shape[0]
    chi2 = A.shape[-1]
    eta1 = M.shape[0]
    eta2 = M.shape[-1]

    Ap = np.zeros((eta1*chi1,2,eta2*chi2), dtype=complex)

    #Test = np.zeros((eta1*chi1,2,eta2*chi2))

    if(debug):
        print(chi1, chi2, eta1, eta2, Ap.shape)

    #for α1 in range(chi1):  # FIX
    #    for α2 in range(chi2):
    #        for β1 in range(eta1):
    #            for β2 in range(eta2):
    #                γ1 = eta1*α1 + β1
    #                γ2 = eta2*α2 + β2
    #                Ap[γ1,:,γ2] = M[β1,:,0,β2]*A[α1,0,α2] + M[β1,:,1,β2]*A[α1,1,α2]

    Ap[:, 0, :] = np.kron(A[:,0,:], M[:,0,0,:]) + np.kron(A[:,1,:], M[:,0,1,:])
    Ap[:, 1, :] = np.kron(A[:,0,:], M[:,1,0,:]) + np.kron(A[:,1,:], M[:,1,1,:])

    #print(Test)
    #print(Ap)

    return Ap

def Apply_MPO(MPO, Train, debug=False, tol=ν, ChiMax=ΧMax, compress=True):
    C = [Apply_MPO_step(MPO[0], Train[0], debug=debug)]
    for i in range(1, len(Train)):
        M = Apply_MPO_step(MPO[i], Train[i], debug=debug)
        C.append(M)
        if compress:
            Compression_step(C, i-1, tol=tol, ChiMax=ChiMax)
            #Sweep(C, tol=tol, ChiMax=ChiMax)
    if not compress:
        return C
    Compression_sweep_left(C, tol=tol, ChiMax=ChiMax)
    return Sweep(C, tol=tol, ChiMax=ChiMax)

def Apply_MPO_MD(MPO_, Train, dim=0, tol=ν, ChiMax=ΧMax, debug=False):

    # make MPO multi dimentinal
    MPO = MD_MPO(MPO_, Train.order, Train.levels, dim=dim)

    # apply the new mpo
    C = [Apply_MPO_step(MPO[0], Train.Train[0], debug=debug)]
    for i in range(1, len(Train.Train)):
        M = Apply_MPO_step(MPO[i], Train.Train[i], debug=debug)
        C.append(M)
        Compression_step(C, i-1, tol=tol, ChiMax=ChiMax)
    Compression_sweep_left(C, tol=tol, ChiMax=ChiMax)

    return TT(Sweep(C, tol=tol, ChiMax=ChiMax), Train.Ns, Train.order, Train.levels, Train.xfs, Train.x0s, Train.tol, Train.ChiMax)




# MPO operations
def Combine_MPO_step(A, B, debug=False):
    chi1, chi3, _, chi2 = A.shape
    eta1, _, eta3, eta2 = B.shape
    if debug:
        print(A.shape, B.shape)

    M = np.tensordot(A, B, axes=([2], [1])).transpose(0, 3, 1, 4, 2, 5).reshape(chi1*eta1, chi3, eta3, chi2*eta2)  # ADDED week2
    #M = np.einsum('ijkl,akbc -> iajblc',A,B, optimize=True).reshape((chi1*eta1, 2, 2, chi2*eta2))
    return M

def Combine_MPO(MPO1, MPO2, tol=ν, ChiMax=ΧMax, debug=False, compress=True):  # Combine 2 MPOs to 1 MPO, (MPO A acts first)
    C = [Combine_MPO_step(MPO1[0], MPO2[0], debug=debug)]
    #n = len(MPO1)
    for i in range(1,len(MPO1)):
        M = Combine_MPO_step(MPO1[i], MPO2[i], debug=debug)
        C.append(M)
        if (compress):
            C = Compression_step_MPO(C, i-1, tol=tol, ChiMax=ChiMax)
    if (compress):
        return Sweep_MPO(C, tol=tol, ChiMax=ChiMax, debug=debug)#, pre='left')
    else:
        return C

def Add_MPO_step(A, B, pre=0, debug=False):
    α1, α2 = A.shape[0], A.shape[-1]
    β1, β2 = B.shape[0], B.shape[-1]

    if pre == -1:  # if both first vector and dimention is 1
        if (α1 == 1 and β1 ==1):
            M = np.zeros((1,2,2,α2+β2), dtype=complex)
            if(debug):
                print(α1, α2, β1, β2)
                print(M.shape, 'Shape inserted 1\n')
            M[0,0,0,:] =  np.hstack([A[0,0,0,:], B[0,0,0,:]])
            M[0,0,1,:] =  np.hstack([A[0,0,1,:], B[0,0,1,:]])
            M[0,1,0,:] =  np.hstack([A[0,1,0,:], B[0,1,0,:]])
            M[0,1,1,:] =  np.hstack([A[0,1,1,:], B[0,1,1,:]])
            return M
    elif pre == 1:  # if both last vector and dimention is 1
        if (α2 == 1 and β2 == 1):
            M = np.zeros((α1+β1,2,2,1), dtype=complex)
            if(debug):
                print(α1, α2, β1, β2)
                print(M.shape, 'Shape inserted 2\n')
            M[:,0,0,0] =  np.hstack([A[:,0,0,0], B[:,0,0,0]])
            M[:,0,1,0] =  np.hstack([A[:,0,1,0], B[:,0,1,0]])
            M[:,1,0,0] =  np.hstack([A[:,1,0,0], B[:,1,0,0]])
            M[:,1,1,0] =  np.hstack([A[:,1,1,0], B[:,1,1,0]])
            return M

    if(debug):
        print(α1, α2, β1, β2)

    M = np.zeros((α1+β1, 2, 2, α2+β2), dtype=complex)

    M[:,0,0,:] =  np.block([
        [A[:,0,0,:], np.zeros((α1,β2))],
        [np.zeros((β1,α2)), B[:,0,0,:]]
    ])
    M[:,0,1,:] =  np.block([
        [A[:,0,1,:], np.zeros((α1,β2))],
        [np.zeros((β1,α2)), B[:,0,1,:]]
    ])
    M[:,1,0,:] =  np.block([
        [A[:,1,0,:], np.zeros((α1,β2))],
        [np.zeros((β1,α2)), B[:,1,0,:]]
    ])
    M[:,1,1,:] =  np.block([
        [A[:,1,1,:], np.zeros((α1,β2))],
        [np.zeros((β1,α2)), B[:,1,1,:]]
    ])

    return M

def Add_MPO(MPO1, MPO2, debug=False, tol=ν, ChiMax=ΧMax):
    new_MPO = []
    N = len(MPO1)

    new_MPO.append(Add_MPO_step(MPO1[0], MPO2[0], pre=-1, debug=debug))

    for (i, (A, B)) in enumerate(zip(MPO1[1:], MPO2[1:])):
        new_MPO.append(Add_MPO_step(A, B, pre=N-1-i, debug=debug))
        new_MPO = Compression_step_MPO(new_MPO, i, tol=tol, ChiMax=ChiMax)

    #return new_MPO
    return Sweep_MPO(new_MPO, tol=tol, ChiMax=ChiMax, pre='left')

def Sub_MPO(MPO1, MPO2, debug=False):
    return Add_MPO(MPO1, Multiply_Const_MPO(MPO2, -1), debug=debug)

def Multiply_Const_MPO(MPO, C):
    new_MPO = copyT(MPO)
    new_MPO[0] = C * new_MPO[0]

    return new_MPO

def Dagger_MPO(MPO):
    return [np.conjugate(M).transpose(0, 2, 1, 3) for M in MPO]

def Unitary(MPO):
    Dag = Dagger_MPO(MPO)

    R = Combine_MPO_step(MPO[0], Dag[0])

    A = R[:, 0, 0, :] + R[:, 1, 1, :]
    for (M, D) in zip(MPO[1::], Dag[1::]):
        R = Combine_MPO_step(M, D)
        A = A @ (R[:, 0, 0, :] + R[:, 1, 1, :])

    return np.trace(A)/(2**len(MPO))

def MD_MPO(MPO_, order, levels, dim):
    MPO = []
    temp = 1

    for (i, cord) in enumerate(order):
        if dim == cord:
            MPO.append(MPO_[levels[i]])
            temp = MPO_[levels[i]].shape[-1]
        else:
            MPO.append(Identity_MPO_step(temp))
    return MPO




# MPOs
def Shift_minus_plus_Iden_MPO_step():
    return np.array([[[[1, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0]]],
                    [[[0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]]],
                    [[[0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 1]]]], dtype=complex)

def Shift_minus_plus_Iden_MPO(N):  # for interpolation of train

    S = Shift_minus_plus_Iden_MPO_step()

    MPO = [(S[0 ,: , : , :] + S[2 ,: , : , :]).reshape(1, 2, 2, -1)]

    for i in range(N-2):
        MPO.append(S)

    MPO.append((S[: ,: , : , 1] + S[: ,: , : , 2]).reshape(-1, 2, 2, 1))

    return MPO

def Twos_comp_MPO_step():
    return np.array([[[[1., 0.], [0., 0.]],
                    [[0., 0.], [1., 0.]]],
                    [[[0., 0.], [0., 1.]],
                    [[0., 1.], [0., 0.]]]])

def Twos_comp_MPO(N):
    A0 = np.zeros((1, 2, 2, 2))
    A0[0, 0, 0, 0] = 1.0
    A0[0, 1, 1, 1] = 1.0

    MPO = [A0]

    A = Twos_comp_MPO_step()

    for i in range(N-2):
        MPO.append(A)

    MPO.append((A[:, :, :, [0]] + A[:, :, :, [1]]))
    return MPO

def MD_TwosComp_MPO(Ns, order, levels, tol=ν, ChiMax=ΧMax):
    MPO = MD_MPO(Flip_MPO(Twos_comp_MPO(Ns[0])), order, levels, 0)
    for i in range(1, len(Ns)):
        MPO = Combine_MPO(MPO, MD_MPO(Flip_MPO(Twos_comp_MPO(Ns[i])), order, levels, i), tol, ChiMax)
    return MPO

def Shift_MPO_minus(N, periodic=False):

    C = np.array([[[[1, 0], [0, 1]], [[0, 0], [1, 0]]],
                    [[[0, 0], [0, 0]], [[0, 1], [0, 0]]]], dtype=complex)

    MPO = [C.copy() for _ in range(N)]
    if periodic:
        MPO[0]  = np.array([C[0,:,:,:]+C[1,:,:,:]])
    else:
        MPO[0]  = np.array([C[0,:,:,:]])

    MPO[-1] = C[:,:,:,1].reshape((2,2,2,1))

    return MPO

def Shift_MPO_plus(N, periodic=False):

    C = np.array([[[[1, 0], [0, 0]], [[0, 1], [1, 0]]],
                    [[[0, 0], [0, 1]], [[0, 0], [0, 0]]]], dtype=complex)

    MPO = [C.copy() for _ in range(N)]
    if periodic:
        MPO[0]  = np.array([C[0,:,:,:]+C[1,:,:,:]])
    else:
        MPO[0]  = np.array([C[0,:,:,:]])

    MPO[-1] = C[:,:,:,1].reshape((2,2,2,1))

    return MPO

def Identity_MPO_step(chi):
    return np.eye(chi, dtype=complex).reshape(chi, 1, 1, chi) * np.eye(2, dtype=complex).reshape(1, 2, 2, 1)

def Identity_MPO(N):  # (1x2x2x1)
    I = np.array([[[[1], [0]], [[0], [1]]]], dtype=complex)
    Iden = [I]

    for i in range(N-1):
        Iden.append(I.copy())

    return Iden


def Derive_MPO_Direct(N, xf=1, x0=0, debug=False, mode='right'):
    if mode == 'right':
        MPO = Multiply_Const_MPO(Sub_MPO(Identity_MPO(N), Shift_MPO_plus(N), debug=debug), ((2**N)/(xf-x0)))
    elif mode == 'center':
        MPO = Multiply_Const_MPO(Sub_MPO(Shift_MPO_minus(N), Shift_MPO_plus(N), debug=debug), ((2**(N-1))/(xf-x0)))
    else:
        MPO = Multiply_Const_MPO(Sub_MPO(Shift_MPO_minus(N), Identity_MPO(N), debug=debug), ((2**N)/(xf-x0)))
    if debug:
        for M in MPO:
            print(M)
            print(' ')
    return MPO

def Derive_MPO(N, xf=1, x0=0, debug=False, mode='center', periodic=False):  # derivative in space (dy/dx) not time (dy/dt)
    MPO = []
    dx = ((2**N)/(xf-x0))
    if mode == 'right':  # right handed
        MPO.append(np.array([[[[dx, -dx, 0], [0, 0, -dx if periodic else 0]], [[0, 0, -dx], [dx, -dx, 0]]]], dtype=complex))

        for i in range(N-2):
            MPO.append(np.array([[[[1, 0, 0], [0, 0, 0]],  [[0, 0, 0], [1, 0, 0]]],
                                 [[[0, 1, 0], [0, 0, 0]],  [[0, 0, 1], [0, 1, 0]]],
                                 [[[0, 0, 0], [0, 0, 1]],  [[0, 0, 0], [0, 0, 0]]]], dtype=complex))
        
        MPO.append(np.array([[[[1], [0]],  [[0], [1]]],
                             [[[0], [0]],  [[1], [0]]],
                             [[[0], [1]],  [[0], [0]]]], dtype=complex))

    elif mode == 'center':
            dx /= 2
            MPO.append(np.array([[[[dx, 0, -dx, 0], [0, dx, 0, -dx if periodic else 0]], [[0, dx if periodic else 0, 0, -dx], [dx, 0, -dx, 0]]]], dtype=complex))
            
            for i in range(N-2):
                MPO.append(np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 0], [1, 0, 0, 0]]],
                                        [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 1, 0, 0], [0, 0, 0, 0]]],
                                        [[[0, 0, 1, 0], [0, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 0]]],
                                        [[[0, 0, 0, 0], [0, 0, 0, 1]], [[0, 0, 0, 0], [0, 0, 0, 0]]]], dtype=complex))
            
            MPO.append(np.array([[[[0], [1]],[[0], [0]]],
                                [[[0], [0]],[[1], [0]]],
                                [[[0], [0]],[[1], [0]]],
                                [[[0], [1]],[[0], [0]]]], dtype=complex))

    else:  # left handed
        MPO.append(np.array([[[[dx, 0, -dx], [0, dx, 0]], [[0, dx if periodic else 0, 0], [dx, 0, -dx]]]], dtype=complex))
        for i in range(N-2):
            MPO.append(np.array([[[[1, 0, 0], [0, 1, 0]],  [[0, 0, 0], [1.0, 0, 0]]],
                                 [[[0, 0, 0], [0, 0, 0]],  [[0, 1, 0], [0, 0, 0]]],
                                 [[[0, 0, 1], [0, 0, 0]],  [[0, 0, 0], [0, 0, 1]]]], dtype=complex))
        
        MPO.append(np.array([[[[0], [1]],  [[0], [0]]],
                             [[[0], [0]],  [[1], [0]]],
                             [[[1], [0]],  [[0], [1]]]], dtype=complex))

    if debug:
        for M in MPO:
            print(M)
            print(' ')
    return MPO

def SecondDerive_MPO2(N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, debug=False, mode='right'):
    if mode == 'right':
        return Combine_MPO(Derive_MPO_Direct(N, xf=xf, x0=x0, debug=debug), Derive_MPO_Direct(N, xf=xf, x0=x0, debug=debug, mode='left'), tol=tol, ChiMax=ChiMax)
    else:
        return Combine_MPO(Derive_MPO_Direct(N, xf=xf, x0=x0, debug=debug, mode='left'), Derive_MPO_Direct(N, xf=xf, x0=x0, debug=debug), tol=tol, ChiMax=ChiMax)

def SecondDerive_MPO3(N, xf=1, x0=0, tol=ν, ChiMax=ΧMax, debug=False, mode='right'):
    if mode == 'right':
        return Combine_MPO(Derive_MPO(N, xf=xf, x0=x0, debug=debug), Derive_MPO(N, xf=xf, x0=x0, debug=debug, mode='left'), tol=tol, ChiMax=ChiMax, compress=False)
    else:
        return Combine_MPO(Derive_MPO(N, xf=xf, x0=x0, debug=debug, mode='left'), Derive_MPO(N, xf=xf, x0=x0, debug=debug), tol=tol, ChiMax=ChiMax, compress=False)


def SecondDerive_MPO(N, xf=1, x0=0, periodic=False):  # Calculate second derivative MPO
    return Multiply_Const_MPO(Add_MPO(Add_MPO(Shift_MPO_plus(N, periodic=periodic), Multiply_Const_MPO(Identity_MPO(N), -2.0)), Shift_MPO_minus(N, periodic=periodic)), (((2**N)/(xf-x0))**2))

def Laplacian_MPO(Ns, xfs, x0s, order, levels, periodic=False):

    if (not isinstance(Ns, Iterable)) or (len(Ns) == 1):  # incase i mess up calling my own functions cuz i know how well i remember stuff....
        Single_DER = Derive_MPO(Ns[0] if isinstance(Ns, Iterable) else Ns, xfs[0] if isinstance(xfs, Iterable) else xfs, x0s[0] if isinstance(x0s, Iterable) else x0s, periodic=periodic)
        return MD_MPO(Combine_MPO(Single_DER, Single_DER), order, levels, 0)

    DERs = []

    LAP = MD_MPO(SecondDerive_MPO(Ns[0], xfs[0], x0s[0], periodic=periodic), order, levels, 0)

    for (i, N) in enumerate(Ns[1::], 1):

        Multi_DER = MD_MPO(SecondDerive_MPO(N, xfs[i], x0s[i], periodic=periodic), order, levels, i)

        LAP = Add_MPO(LAP, Multi_DER)

    return LAP

def Fourier_H():
    temp = 1/np.sqrt(2)
    return np.array([[[[temp, 0], [temp, 0]],
                      [[0, temp], [0, -temp]]]], dtype=complex)

def Fourier_Aide(n, C):  # R matrix
    R = np.zeros((2, 2, 2, 2), dtype=complex)
    for a in (0, 1):  # s
        for b in (0, 1):  # s'
            for c in (0, 1):  # x
                for d in (0, 1):  # k
                    if (a==b) and (c==d):
                        R[a, d, c, b] = np.exp(C*1j*np.pi*a*c/(2**n))  # s,x,k,s'  2j not used since considered in 2**n
    return R

def Fourier_Layer(N, C, i):  # layering without identity fill

    fi = []

    fi.append(Fourier_H()) #[i]

    for j in range(i+1, N):
        fi.append(Fourier_Aide(j-i, C)) #[j]


    sh = (fi[-1].shape[:3])
    fi[-1] = np.sum(fi[-1], -1).reshape(*sh, 1)

    return fi  # returns array of length N-i

def Fourier_combine(A, B, R, tol=ν, ChiMax=ΧMax, debug=False, compress1=True):  # assume A is longer then B

    Comb = Combine_MPO(B, A[(R+1):], compress=compress1)  # doesnt need to compress, changes nothing

    if debug:
        Train_Shape(A[(R+1):])
        Train_Shape(B)
        print(B)
        print(A[(R+1):], "brk\n")
        Train_Shape(Comb)
        print('ok')

    New_Four = A[:R+1] + Comb

    #if (compress2):  # confirm if needs to be here?
    #    Compression_step_MPO(New_Four, R, ChiMax=ΧMax, tol=ν)  # get rid of? compressing already in combine_MPO funciton
    # final note: got rid of because no reason to compress a simpler etensor with the added rows of previous tensors, the 'A[:R+1]' should already be as compressed as possible
    # this was only done to try and replicate a the paper that shows the compression steps

    if compress1:
        New_Four = Sweep_MPO(New_Four, tol=tol, ChiMax=ChiMax)


    if debug:
        print("New MPO shape is")
        Train_Shape(New_Four, R)

    return New_Four

def Fourier_MPO(N, tol=ν, ChiMax=ΧMax, inv=False, debug=False, debug_=False, force=False, compress=True, compress3=True):

    if not(force):
        try:
            if(N > 5 and N < 100 and ChiMax<=1000 and tol>=1e-25 and compress==True):
                if(inv):
                    loaded = np.load(os.path.join("Fourier_saved", f"invFourier_{N}.npz"))
                else:
                    loaded = np.load(os.path.join("Fourier_saved", f"Fourier_{N}.npz"))
                data = loaded['arr_0']
                Bond = loaded['arr_1']
                N = loaded['arr_2']

                Bond = [int(b) for b in Bond]
                start = 0
                MPO = []
                if(True):
                    print("loaded: ", inv)

                for i in range(int(N)):
                    #print(i, Bond[i]*4*Bond[i+1])
                    MPO.append(np.reshape(data[start:start+4*Bond[i]*Bond[i+1]:], (Bond[i], 2, 2, Bond[i+1])))
                    start += 4*Bond[i]*Bond[i+1]

                return MPO
        except:
            print("Failed to load fourier MPO")

    if inv:
        C = -1
    else:
        C = 1

    MPO = Fourier_combine(Fourier_Layer(N, C, 0), Fourier_Layer(N, C, 1), 0, tol=tol, ChiMax=ChiMax, compress1=compress3)
    for i in range(1, N-1):
        MPO = Fourier_combine(MPO, Fourier_Layer(N, C, i+1), i, tol=tol, ChiMax=ChiMax, debug=debug, compress1=compress3)

    if compress:  # left sweep after everything else
        MPO = Compression_sweep_left_MPO(MPO, tol, ChiMax)  # the only needed compression... sadly...  (will add another because needs to be faster, figuring out which is best to add and where)

    if(debug):
        print(1/np.sqrt(2**N))

    if(inv):
        return Flip_MPO(MPO)
    else:
        return MPO

def MD_Fourier_MPO(Ns, order, level, tol=ν, ChiMax=ΧMax, inv=False):

    MPO = MD_MPO(Fourier_MPO(Ns[0], tol=tol, ChiMax=ChiMax, inv=inv), order, level, 0)

    for i in range(1, len(Ns)):
        MPO = Combine_MPO(MPO, MD_MPO(Fourier_MPO(Ns[i], tol=tol, ChiMax=ChiMax, inv=inv), order, level, i), tol, ChiMax)

    return MPO

def Multiply_MPO(Train, tol=ν, ChiMax=ΧMax):
    Train = Sweep(Train, tol=tol, ChiMax=ChiMax)
    MPO = []
    for T in Train:
        Chi1, _, Chi2 = T.shape

        M = np.zeros((Chi1, 2, 2, Chi2), dtype=complex)
        M[:,0,0,:] = T[:,0,:]
        M[:,1,1,:] = T[:,1,:]
        MPO.append(M)
    #print('multi')
    return Sweep_MPO(MPO, tol=tol, ChiMax=ChiMax)

#Schrodinger MPO's
def Schrodinger_MPO(potential, xf, x0, N, dt, FFT, iFFT, cut = 50, tol=1e-25, ChiMax = 1000, debug=False):
    
    def Vx(x):  # potential
        return np.exp(-0.5j*(dt)*potential(x))
    
    # Both functions in Trains
    V = Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
    T = Chop_Train(Flip_Train(Func_to_Train_Custum_Pot(dt, N, xf=(2**(N-1))*((2*np.pi)/(xf-x0)), x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)  # slowing step, keep eye on this

    # Making them multiplication MPOs
    V_MPO = Multiply_MPO(V, tol=tol, ChiMax=ChiMax)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)

    print('1:', Max_Bond(V_MPO))
    print('2:', Max_Bond(T_MPO))
    # Making main MPO
    MPO = Combine_MPO(FFT, V_MPO, tol=tol, ChiMax=ChiMax)
    print('3:', Max_Bond(MPO))
    MPO = Combine_MPO(T_MPO, MPO, tol=tol, ChiMax=ChiMax)
    print('4:', Max_Bond(MPO))
    MPO = Combine_MPO(iFFT, MPO, tol=tol, ChiMax=ChiMax)  # slow step
    print('5:', Max_Bond(MPO))
    MPO = Combine_MPO(V_MPO, MPO, tol=tol, ChiMax=ChiMax)
    print('6:', Max_Bond(MPO))

    if(debug):
        Train_Shape(MPO)

    return MPO

def Schrodinger_MPO_sigmoid(potential, xf, x0, N, dt, FFT, iFFT, cut = 50, tol=1e-25, ChiMax = 1000, debug=False):

    def Vx(x):  # potential
        return np.exp(-0.5j*(dt)*potential(x))

    def Tk(k):  # kinetic
        return np.exp(-0.5j*(k**2)*(dt)) * sp.special.expit(k+cut) * sp.special.expit(cut-k)

    # Both functions in Trains
    V = Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
    T = Chop_Train(Flip_Train(Func_to_Train_Custum(Tk, N, xf=(2**(N-1))*((2*np.pi)/(xf-x0)), x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)  # slowing step, keep eye on this

    # Making them multiplication MPOs
    V_MPO = Multiply_MPO(V, tol=tol, ChiMax=ChiMax)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)

    print('1: ', Max_Bond(V_MPO))
    print('2: ', Max_Bond(T_MPO))
    # Making main MPO
    MPO = Combine_MPO(FFT, V_MPO, tol=tol, ChiMax=ChiMax)
    print('3: ', Max_Bond(MPO))
    MPO = Combine_MPO(T_MPO, MPO, tol=tol, ChiMax=ChiMax)
    print('4: ', Max_Bond(MPO))
    MPO = Combine_MPO(iFFT, MPO, tol=tol, ChiMax=ChiMax)
    print('5: ', Max_Bond(MPO))
    MPO = Combine_MPO(V_MPO, MPO, tol=tol, ChiMax=ChiMax)
    print('6: ', Max_Bond(MPO))
    return MPO




# Compression

# Trains
def Compression_step(Train, i, tol=ν, ChiMax=ΧMax, mode='right', debug=False):
    A = Train[i] # a,s,b
    B = Train[i+1] # a,s,b

    chi1, chi2 = A.shape[0], B.shape[-1]
    te1, te2 = A.shape[1], B.shape[1]


    M = np.tensordot(A, B, axes=([2], [0])).reshape((te1*chi1, te2*chi2))  # ADDED week2

    try:  # ADDED week2
        try:  # fast path — gesdd is scipy's default divide-and-conquer driver
            U, σ, W = sp.linalg.svd(M, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesdd')
        except Exception as e:
            print(f"error: {e}")
            print("gesdd failed to converge, falling back to gesvd")
            U, σ, W = sp.linalg.svd(M, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
    except Exception as e:
        print(f"error: {e}")
        print("SVD FAILED!! UH OH!! (maybe inf or nan, maybe no SVDable)")
        return Train

    err = np.append(np.cumsum(σ[::-1]**2)[::-1], 0)
    max_err = tol*err[0]
    Chi_cut = np.argmax(err[1::] < max_err)  # can give 0 if all true (not to worry anymore, only keep in mind )
    Chi = min(ChiMax, Chi_cut+1)

    if mode == 'right':
        W = σ[:Chi, None] * W[:Chi, :]
        U = U[:, :Chi]
    else:
        U = U[:, :Chi] * σ[None, :Chi]
        W = W[:Chi, :]
    
    Train[i]   = U.reshape(chi1, te1, Chi)
    Train[i+1] = W.reshape(Chi, te2, chi2)


    return Train

def Compression_sweep_right(Train, tol=ν, ChiMax=ΧMax, debug=False):
    bonds.append(Max_Bond(Train))
    for i in range(len(Train)-1):
        Compression_step(Train,i,tol=tol,ChiMax=ChiMax,mode='right')
    return Train

def Compression_sweep_left(Train, tol=ν, ChiMax=ΧMax, debug=False):
    bonds.append(Max_Bond(Train))
    for i in range(len(Train)-2, -1, -1):
        Compression_step(Train,i,tol=tol,ChiMax=ChiMax,mode='left')
    return Train

def Sweep(Train, N=20, tol=ν, ChiMax=ΧMax, debug=False):  # replaces input train with the compressed train
    

    Χ = Max_Bond(Train)  # ADDED week2
    og = Χ
    for i in range(N):
        Compression_sweep_right(Train, tol=tol, ChiMax=ChiMax, debug=debug)
        Compression_sweep_left(Train, tol=tol, ChiMax=ChiMax, debug=debug)
        temp = Max_Bond(Train)
        if(Χ == temp):  # ADDED week2, return if compression is no longer helping
            Compression_sweep_right(Train, tol=tol, ChiMax=ChiMax, debug=debug)  # week 3, to make sure always right canonical form

            #Compression_sweep_left(Train, tol=tol, ChiMax=ChiMax, debug=debug)
            #Compression_sweep_right(Train, tol=tol, ChiMax=ChiMax, debug=debug)  # one extra back and fourth, determined not needed after some runs where the flag never popped
            if debug == 1:
                Train_Shape(Train)
            if Max_Bond(Train) < temp:
                print('Sweep didnt cap!!\n')
            #print('broke at loop ', i, " with bond ", temp, ", original bond ", og, " and backup ", Max_Bond(Train))
            return Train
        else:
            Χ = temp
    Compression_sweep_right(Train, tol=tol, ChiMax=ChiMax, debug=debug)  # week 3
    #print('Final bond is ', Χ, ", original bond was ", og, " and backup ", Max_Bond(Train))
    return Train

# MPO
def Compression_step_MPO(MPO, i, tol=ν, ChiMax=ΧMax, mode='right', debug=False):  # FIX make it only convert the nessisary tensors
    AB = MPO[i:i+2]
    M = MPS_to_MPO(Compression_step(MPO_to_MPS(AB), 0, tol, ChiMax, mode, debug))
    MPO[i:i+2] = M

    return MPO

def Compression_sweep_right_MPO(MPO, tol=ν, ChiMax=ΧMax, debug=False):
    return MPS_to_MPO(Compression_sweep_right(MPO_to_MPS(MPO), tol, ChiMax, debug))

def Compression_sweep_left_MPO(MPO, tol=ν, ChiMax=ΧMax, debug=False):
    return MPS_to_MPO(Compression_sweep_left(MPO_to_MPS(MPO), tol, ChiMax, debug))

def Sweep_MPO(MPO, N=20, tol=ν, ChiMax=ΧMax, debug=False, pre='right'):  # cant replace input MPO since it changes it to a MPS
    if (pre=='left'):  # for MPOs that are compressed left first
        #print('Max bond for left:', Max_Bond(MPO))
        MPS = Compression_sweep_left(MPO_to_MPS(MPO), tol, ChiMax, debug)
        return MPS_to_MPO(Sweep(MPS, N, tol, ChiMax, debug))
    else:
        #print("Test")
        return MPS_to_MPO(Sweep(MPO_to_MPS(MPO, debug), N, tol, ChiMax, debug), debug)




# Conversion HELP in next version fix to not need new lists for the conversion

def MPO_to_MPS(MPO, debug=False):
    MPS = []
    for M in MPO:
        a,i,j,b = M.shape
        MPS.append(M.reshape(a,i*j,b))
    if debug:
        print('into MPS')
        Train_Shape(MPS)
    return MPS

def MPS_to_MPO(MPS, debug=False, mode='right'):
    MPO = []
    k = MPS[0].shape[1]
    if k == 2:
        if mode=='right':
            for M in MPS:
                a,k,b = M.shape
                MPO.append(M.reshape(a, 1, 2, b))
        elif mode=='left':
            for M in MPS:
                a,k,b = M.shape
                MPO.append(M.reshape(a, 2, 1, b))
    elif k == 4:
        for M in MPS:
            a,k,b = M.shape
            MPO.append(M.reshape(a, 2, 2, b))
    elif k == 1:
        for M in MPS:
            a,k,b = M.shape
            MPO.append(M.reshape(a, 1, 1, b))
    else:
        print(k)
        print('Transition error into MPO')

    if debug:
        print('back to MPO')
        Train_Shape(MPO)

    return MPO




# Conventinal methods
def Conven_Momentum(psi, k_p, dx):  # calculates with fourier
    dpsi = sp.fftpack.ifft(1j * k_p * sp.fftpack.fft(psi))
    rho_list = np.imag(np.conjugate(psi) * dpsi)
    return np.sum(rho_list)*dx  # average current

def Conven_Momentum2(psi, dx):  # calculates in real space
    dpsi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2*dx)
    rho_list = np.imag(np.conjugate(psi) * dpsi)  # probability current
    return np.sum(rho_list)*dx  # average current

def Conven_Norm(psi, dx):  # conventinal calculation for norm
    return np.sum(np.abs(psi)**2)*dx




# Evolutions
def Schrod_evo1(N, initial, potential, xf, x0, tf, dt, tol=1e-25, ChiMax=1000, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')
    MPO = Schrodinger_MPO(potential, x0=x0, xf=xf, N=N, dt=dt, FFT=FFT, iFFT=iFFT, cut=cut, tol=tol)

    n = 2**N
    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)

    output_n = np.zeros((M), dtype=complex)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)

    for (i, t) in enumerate(t_points[1::], 1):
        psi = Apply_MPO(MPO, psi, tol=tol, ChiMax=ChiMax)
        output_rho[i] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i] = Norm_Train(psi, xf=xf, x0=x0)
    return psi, output_n, output_rho, Max_Bond(MPO)

def Schrod_evo2(N, initial, potential, xf, x0, tf, dt, tol=1e-25, ChiMax=1000, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')
    MPO = Schrodinger_MPO_sigmoid(potential, x0=x0, xf=xf, N=N, dt=dt, FFT=FFT, iFFT=iFFT, cut=cut, tol=tol)

    n = 2**N
    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)

    output_n = np.zeros((M), dtype=complex)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)

    for (i, t) in enumerate(t_points[1::], 1):
        psi = Apply_MPO(MPO, psi, tol=tol, ChiMax=ChiMax)
        output_rho[i] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i] = Norm_Train(psi, xf=xf, x0=x0)
    return psi, output_n, output_rho, Max_Bond(MPO)

def Schrod_evo_Apply(N, initial, potential, xf, x0, tf, dt, tol=1e-25, ChiMax=1000, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')

    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)

    output_n = np.zeros((M), dtype=complex)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)

    def Vx(x):  # potential
        return np.exp(-0.5j*(dt)*potential(x))
    
    # Both functions in Trains
    V = Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
    T = Chop_Train(Flip_Train(Func_to_Train_Custum_Pot(dt, N, xf=(2**(N-1))*((2*np.pi)/(xf-x0)), x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)  # slowing step, keep eye on this

    # Making them multiplication MPOs
    V_MPO = Multiply_MPO(V, tol=tol, ChiMax=ChiMax)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)


    for (i, t) in enumerate(t_points[1::], 1):
        if i%100 == 0:
            print(t)
        
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(FFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(T_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(iFFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)

        output_rho[i] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i] = Norm_Train(psi, xf=xf, x0=x0)
    
    return psi, output_n, output_rho, np.max([Max_Bond(V_MPO), Max_Bond(FFT), Max_Bond(T_MPO), Max_Bond(iFFT)])

def Schrod_evo_Apply2(N, initial, potential, xf, x0, tf, dt, tol=1e-25, ChiMax=1000, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')

    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)

    output_n = np.zeros((M), dtype=complex)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)

    def Vx(x):  # potential
        return np.exp(-0.5j*(dt)*potential(x))
    
    def Tk(k):
        return np.exp(-0.5j*(k**2)*dt) * sp.special.expit(k+cut) * sp.special.expit(cut-k)
    
    # Both functions in Trains
    V = Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax)
    T = Chop_Train(Flip_Train(Func_to_Train_Custum(Tk, N, xf=(2**(N-1))*((2*np.pi)/(xf-x0)), x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)  # slowing step, keep eye on this

    # Making them multiplication MPOs
    V_MPO = Multiply_MPO(V, tol=tol, ChiMax=ChiMax)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)


    for (i, t) in enumerate(t_points[1::], 1):
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(FFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(T_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(iFFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)

        output_rho[i] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i] = Norm_Train(psi, xf=xf, x0=x0)
    
    return psi, output_n, output_rho, np.max([Max_Bond(V_MPO), Max_Bond(FFT), Max_Bond(T_MPO), Max_Bond(iFFT)])

def Comp_Schrod_evo(N, initial, potential, xf, x0, tf, dt, factor=1):


    def Vx(x):  # potential
        return np.exp(-0.5j*(dt/factor)*potential(x))

    def Tk(k):  # kinetic
        return np.exp(-0.5j*(k**2)*(dt/factor))

    n = 2**N
    dx = (xf-x0)/n
    t_points = np.arange(0, tf, dt)
    x_points = np.arange(x0, xf, dx)

    M = len(t_points)

    dk = (2*np.pi)/(n*dx)
    k_points = -0.5*(n)*dk + dk*np.arange(n) # + dk/2
    k_points = np.append(k_points[2**(N-1)::], k_points[:2**(N-1):])

    ψ = initial(x_points)

    V = np.array([Vx(x) for x in x_points])
    T = np.array([Tk(k) for k in k_points])


    output_n = np.zeros((M), dtype=complex)
    output_rho = np.zeros((M), dtype=complex)

    output_n[0] = Conven_Norm(ψ, dx)
    output_rho[0] = Conven_Momentum(ψ, k_points, dx)


    for (i, t) in enumerate(t_points[1::], 1):
        if i%100 == 0:
            print(t)

        for j in range(factor):
            ψ = ψ * V
            ψ_k = sp.fftpack.fft(ψ)
            ψ_k = ψ_k * T
            ψ = sp.fftpack.ifft(ψ_k)
            ψ = ψ * V
        
        output_n[i] = Conven_Norm(ψ, dx)
        output_rho[i] = Conven_Momentum(ψ, k_points, dx)

    return ψ, output_n, output_rho, 1

# Changing vector potential
def Schrod_B_evo_change(N, initial, potential, vec_pot, xf, x0, tf, dt, tol=1e-25, ChiMax=ΧMax, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)
    
    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    def Vx(x):  # potential
        return np.exp(-0.5j*(dt)*potential(x))
    
    V_MPO = Multiply_MPO(Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax)
    

    output_n = np.zeros((M), dtype=complex)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)

    # Making them multiplication MPOs
    k = (2**(N-1))*((2*np.pi)/(xf-x0))
    T = Chop_Train(Flip_Train(Func_to_Train_Custum_Pot(dt, N, xf=k, x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)

    
    for (i, t) in enumerate(t_points[:-1:], 0):

        B_MPO = Multiply_MPO(Chop_Train(Flip_Train(Exp_Train(N, α=1j*dt*vec_pot[0] * vec_pot[1](t+dt/2), xf=k, x0=-k, shift=0)), N-1), tol=tol, ChiMax=ChiMax)

        if i%100 == 0:
            print(t)
            Train_Shape(psi)

        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(FFT, psi, tol=tol, ChiMax=ChiMax)

        psi = Apply_MPO(T_MPO, psi, tol=tol, ChiMax=ChiMax)

        psi  = Apply_MPO(B_MPO, psi, tol=tol, ChiMax=ChiMax)

        psi = Apply_MPO(iFFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)

        output_rho[i+1] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i+1] = Norm_Train(psi, xf=xf, x0=x0)
    
    return psi, output_n, output_rho, t_points, np.max([np.max([Max_Bond(V_MPO), Max_Bond(FFT), Max_Bond(T_MPO), Max_Bond(iFFT), Max_Bond(B_MPO)])])

def Schrod_B_evo_change2(N, initial, potential, vec_pot, xf, x0, tf, dt, tol=1e-25, ChiMax=ΧMax, cut=25):

    FFT = Fourier_MPO(N, ChiMax, tol)
    iFFT = Fourier_MPO(N, ChiMax, tol, inv=True)
    DER = Derive_MPO_Direct(N, xf=xf, x0=x0, debug=False, mode='center')

    psi = Func_to_Train_Fourier(initial, N, xf=xf, x0=x0, tol=tol)
    
    t_points = np.arange(0, tf, dt)
    M = len(t_points)
    
    def Vx(x):  # potential
        return np.exp(-0.5j*dt*potential(x))
    
    
    V_MPO = Multiply_MPO(Func_to_Train_cmps(Vx, N, xf=xf, x0=x0, tol=tol, ChiMax=ChiMax), tol=tol, ChiMax=ChiMax)
    

    output_n = np.zeros((M), dtype=complex)
    output_rho = np.zeros((M), dtype=complex)
    output_rho[0] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
    output_n[0] = Norm_Train(psi, xf=xf, x0=x0)

    def Tk(k):
        return np.exp(-0.5j*(k**2)*dt) * sp.special.expit(k+cut) * sp.special.expit(cut-k)

    # Making them multiplication MPOs
    k = (2**(N-1))*((2*np.pi)/(xf-x0))
    T = T = Chop_Train(Flip_Train(Func_to_Train_Custum(Tk, N, xf=k, x_cutoff=cut, tol=tol, ChiMax=ChiMax)), N-1)
    T_MPO = Multiply_MPO(T, tol=tol, ChiMax=ChiMax)

    
    for (i, t) in enumerate(t_points[:-1:], 0):

        B_MPO = Multiply_MPO(Chop_Train(Flip_Train(Exp_Train(N, α=1j*dt*vec_pot[0] * vec_pot[1](t+dt/2), xf=k, x0=-k, shift=0)), N-1), tol=tol, ChiMax=ChiMax)

        if i%100 == 0:
            print(t)
            Train_Shape(psi)

        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(FFT, psi, tol=tol, ChiMax=ChiMax)

        psi = Apply_MPO(T_MPO, psi, tol=tol, ChiMax=ChiMax)

        psi  = Apply_MPO(B_MPO, psi, tol=tol, ChiMax=ChiMax)

        psi = Apply_MPO(iFFT, psi, tol=tol, ChiMax=ChiMax)
        psi = Apply_MPO(V_MPO, psi, tol=tol, ChiMax=ChiMax)

        output_rho[i+1] = Momentum(psi, xf=xf, x0=x0, ChiMax=ChiMax, tol=tol, DER=DER)
        output_n[i+1] = Norm_Train(psi, xf=xf, x0=x0)
    
    return psi, output_n, output_rho, t_points, np.max([np.max([Max_Bond(V_MPO), Max_Bond(FFT), Max_Bond(T_MPO), Max_Bond(iFFT), Max_Bond(B_MPO)])])

def Comp_Schrod_B_evo_change(N, initial, potential, vec_pot, xf, x0, tf, dt, factor=1):
    
    if factor <=0:
        factor = 1

    dt = dt/factor

    n = 2**N
    dx = (xf-x0)/n
    x_points = np.arange(x0, xf, dx)

    dk = (2*np.pi)/(n*dx)
    k_points = -0.5*(n)*dk + dk*np.arange(n) # + dk/2
    k_points = np.append(k_points[2**(N-1)::], k_points[:2**(N-1):])

    ψ = initial(x_points)
    ψ_temp = ψ

    t_points = np.arange(0, tf, dt)
    M = len(t_points)

    def Vx(x):  # potential
        return np.exp(-1.0j*(dt/2)*potential(x))

    def Tk(k):  # kinetic
        return np.exp(-0.5j*(k**2)*dt)


    V = np.array([Vx(x) for x in x_points])

    T = np.array([Tk(k) for k in k_points])

    fac_t = len(t_points[::factor])
    output_n = np.zeros(fac_t, dtype=complex)
    output_rho = np.zeros(fac_t, dtype=complex)
    output_n[0] = Conven_Norm(ψ, dx)
    output_rho[0] = Conven_Momentum(ψ, k_points, dx)


    for (i, t) in enumerate(t_points[:-1:], 1):

        A_val = vec_pot[0] * vec_pot[1](t + dt/2)
        B = np.exp(-1j*dt*A_val*k_points)

        if i%100 == 0:
            print(t)

        ψ = ψ * V
        ψ_k = sp.fftpack.fft(ψ)
        ψ_k = ψ_k * T

        ψ_k = ψ_k * B

        ψ = sp.fftpack.ifft(ψ_k)
        ψ = ψ * V
        
        if i % factor == 0:
            ψ_temp = ψ
            output_n[i//factor] = Conven_Norm(ψ, dx)
            output_rho[i//factor] = Conven_Momentum(ψ, k_points, dx)

    return ψ_temp, output_n, output_rho, t_points[::factor], 1

# 2d Wave equation
def Wave(Ns, Train, DER_dt2, xfs, x0s, tf, dt, tol=1e-25):

    Train_dt = Zero(np.sum(Ns))  # initial velocity du/dt = 0
    t_points = np.arange(0, tf, dt)

    for (i, t) in enumerate(t_points[1::], 1):

        Lap_u_dt2 = Apply_MPO(DER_dt2, Train.Train, tol=tol)
        Train_dt = Add_Trains(Train_dt, Lap_u_dt2, tol=tol)                # v_new = v + dt * Laplacian(u)
        Train.Train = Add_Trains(Train.Train, Train_dt, tol=tol)     # u_new = u + dt * v_new




# from seemps but simplified
def qft_mpo(N, sign=1, **kwargs):

    def fix_last(mpo_list):
        A = mpo_list[-1]
        shape = A.shape
        A = np.sum(A, -1).reshape(shape[0], shape[1], shape[2], 1)
        return mpo_list[:-1] + [A]

    # Tensor doing nothing
    noop = np.eye(2).reshape(1, 2, 2, 1)
    #
    # Beginning Hadamard
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)
    Hop = np.zeros((1, 2, 2, 2))
    Hop[0, 1, :, 1] = H[1, :]
    Hop[0, 0, :, 0] = H[0, :]
    #
    # Conditional rotations
    R0 = np.zeros((2, 2, 2, 2))
    R0[0, 0, 0, 0] = 1.0
    R0[0, 1, 1, 0] = 1.0
    R0[1, 0, 0, 1] = 1.0
    R1 = np.zeros((2, 2, 2, 2))
    R1[1, 1, 1, 1] = 1.0
    jϕ = sign * 1j * np.pi
    rots = [R0 + R1 * np.exp(jϕ / (2**n)) for n in range(1, N)]
    #
    return [fix_last([noop] * n + [Hop] + rots[: N - n - 1]) for n in range(0, N)]