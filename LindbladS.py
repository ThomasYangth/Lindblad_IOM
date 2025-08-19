# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function, division

import numpy as np
from scipy.sparse.linalg import eigs as sparseeig
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from time import time
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse

import itertools

from OPlib import *

#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)


######
# NPZ_PATH is the path to save the npz files.
# The npz files will be large, so it is recommended to set it to a scratch directory if on a cluster.
######
NPZ_PATH = "." # os.getcwd().replace("/home","/scratch/gpfs") 
def npz_path (filename):
    return NPZ_PATH + "/" + filename + ".npz"

RCPARAMS = {'font.size': 12, 'mathtext.fontset':'cm'}

######
## The following part are based on examples from QuSpin
## https://quspin.github.io/QuSpin/examples/example23.html
######
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32,count_particles_sig_32 # user basis data types signatures
from quspin.tools.Floquet import Floquet
from quspin.tools.evolution import expm_multiply_parallel
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32,float64,complex128 # numba data types

sps = 4 # 4 states per site

from time import time

def sparsesvd (A, k=1, ncv=None, **kwargs):
    L = np.shape(A)[0]
    ncv0 = min(L, 50)
    ncv = 0
    while True:
        ncv += ncv0
        if ncv > L/2:
            print("Sparse SVD not working, swithcing to dense.")
            return np.linalg.svd(A)
        try:
            u,s,vt = svds(A, k=k, ncv=ncv, **kwargs)
            if len(s) == k:
                return u,s,vt
            else:
                print("SVD with ncv =", ncv, " converged", len(s), "/", k, "singular values.")
        except ArpackNoConvergence:
            print("SVD not converging with ncv = ", ncv)

def fprint (*cont):
    print(*cont, flush=True)

######  function to call when applying operators
@cfunc(op_sig_32,
	locals=dict(b=uint32,occ=int32,sps=uint32,me=complex128), )
def op(op_struct_ptr,op_str,site_ind,N,args):
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	sps = 4

	site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
	occ = (op_struct.state//sps**site_ind)%sps # occupation
	b = sps**site_ind

	# This functions defines the matrix element by modifying op_struct.state and op_struct.matrix_ele
	# The origin op_struct.state is the j in O_{ij}, and the modified is i
	# Since we are doing this on a site, what we actually do is op_struct.state += (i-j)*b

	# For operators:
	# 'X' is for left-multiplication of X, 'x' for right-multiplication of X, etc.
	# 'L' for depolarizing noise, 'H' for dephasing noise, 'C' and 'c' for the two parts of decay noise
	# occ convention: I=0, X=1, Y=2, Z=3
	
	# We will define two arrays, Is and Ms, both contains 4 elements,
	# indicating the corresponding i and matrix_ele for four possible j (or occ).

	Ms = [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
	Is = [0, 1, 2, 3]

	if op_str == ord('X'):
		Ms = [1.+0.j, 1.+0.j, 1.j, -1.j]
		Is = [1, 0, 3, 2]

	elif op_str == ord('x'):
		Ms = [1.+0.j, 1.+0.j, -1.j, 1.j]
		Is = [1, 0, 3, 2]

	elif op_str == ord('Y'):
		Ms = [1.+0.j, -1.j, 1.+0.j, 1.j]
		Is = [2, 3, 0, 1]

	elif op_str == ord('y'):
		Ms = [1.+0.j, 1.j, 1.+0.j, -1.j]
		Is = [2, 3, 0, 1]

	elif op_str == ord('Z'):
		Ms = [1.+0.j, 1.j, -1.j, 1.+0.j]
		Is = [3, 2, 1, 0]

	elif op_str == ord('z'):
		Ms = [1.+0.j, -1.j, 1.j, 1.+0.j]
		Is = [3, 2, 1, 0]

	elif op_str == ord('I') or op_str == ord("i"):
		Ms = [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]
		Is = [0, 1, 2, 3]

	elif op_str == ord('L'):
		Ms = [0.+0.j, -1.+0.j, -1.+0.j, -1.+0.j]
		Is = [0, 1, 2, 3]

	elif op_str == ord('H'):
		Ms = [0.+0.j, -1.+0.j, -1.+0.j, 0.+0.j]
		Is = [0, 1, 2, 3]

	elif op_str == ord('C'):
		Ms = [0.+0.j, -1.+0.j, -1.+0.j, -2.+0.j]
		Is = [0, 1, 2, 3]

	elif op_str == ord('c'):
		Ms = [0.+0.j, 0.+0.j, 0.+0.j, -2.+0.j]
		Is = [0, 1, 2, 0]

	else:
		err = -1
		return err

	op_struct.state += b*(Is[occ]-occ)
	me = Ms[occ]
	op_struct.matrix_ele *= me

	return err
#
op_args=np.array([sps],dtype=np.uint32)

#
######  define symmetry maps
#
@cfunc(map_sig_32,
	locals=dict(shift=uint32,out=uint32,sps=uint32,i=int32,j=int32,) )
def translation(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0
	shift = args[0]
	sps = args[1]
	for i in range(N):
		j = (i+shift+N)%N
		out += ( x%sps ) * sps**j
		x //= sps
	#
	return out
T_args=np.array([1,sps],dtype=np.uint32)

#
@cfunc(map_sig_32,
	locals=dict(out=uint32,sps=uint32,i=int32,j=int32) )
def parity(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0
	sps = args[0]
	for i in range(N):
		j = (N-1) - i
		out += ( x%sps ) * (sps**j)
		x //= sps
	#
	return out
P_args=np.array([sps],dtype=np.uint32)

#
@cfunc(map_sig_32,
	locals=dict(out=uint32,sps=uint32,i=int32) )
def inversion(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0

	sps = args[0]
	for i in range(N):
		out += ( sps-x%sps-1 ) * (sps**i)
		x //= sps
	#
	return out
Z_args=np.array([sps],dtype=np.uint32)

def cyclic_order (s, l=None):
    if l is None:
        l = len(s)
    for i in range(1, l):
        if l%i != 0:
            continue
        if s == s[i:]+s[:i]:
            return i
    return l
    
class Lindbladian:

    def __init__ (self, Hterms, Lterms, L:int, k:int = None, conserved_quantities = [], name = "", use_sparse = False, Hterms_len:int=0, Floquet_t = 0, pbc=True):
        """
        Initializes the Lindbladian object.

        Args:
            Hterms (list): A list of the terms in the Hamiltonian. Each element in the list should have the form
                [float, str], where the string, like "X" or "ZZ", indicates a local Pauli string term, and the float
                indicates the amplitude of that term. If Hterms_len > 0, Hterms would be a list of several Hamiltonians,
                where each Hamiltonian is a list as described above, and the number of Hamiltonians is Hterms_len.
            Lterms (list): A list of the terms in the dissipative part, i.e., jump operators.
            L (int): size of the system
            k (int): Default is None, indicating diagonalizing in the full Hilbert space.
                If given, restrict the diagonalization in the translation block with momentum 2pi*k/L.
            Floquet_t (float): Default is 0, indicates not using Floquet. If non-zero, the matrix consturcted would be a Floquet
                unitary of exp(L*t)exp(-i*Hn*t)...exp(-i*H1*t).
        """
        self.Hterms = Hterms
        self.Lterms = Lterms
        self.L = L
        self.k = k
        self.iom = conserved_quantities
        self.name = name
        self.sparse = use_sparse
        self.hasmat = False
        self.Htl = Hterms_len
        self.FlqT = Floquet_t
        if Hterms_len > 0:
            if Hterms_len == 1:
                if len(self.Hterms) != 1:
                    self.Hterms = [self.Hterms]
            else:
                if len(self.Hterms) != Hterms_len:
                    raise Exception(f"Hterms_len given as {Hterms_len}, but length of Hterms is {len(Hterms)}.")
        self.pbc = pbc
        if not self.pbc:
            if self.k is not None:
                fprint("Translational sector not available for OBC! k is set to None.")
                self.k = None
        
    def construct_mat (self, override=False):

        if override:
            self.hasmat = False

        if self.hasmat:
            return

        t1 = time()

        ######  construct user_basis
        # define maps dict
        if self.k is None:
            # Do not use translational blocks
            maps = dict()
        else:
            maps = dict(T_block=(translation, self.L, self.k, T_args))
        # define op dict
        op_dict = dict(op=op,op_args=op_args)
        # create user basis
        basis = user_basis(np.uint32, self.L, op_dict, allowed_ops=set("XxYyZzIiLHCc"), sps=sps, **maps)
        fprint(f"Basis size: {basis.Ns}")

        ############   create Hamiltonian   #############

        Hs = []

        for Ham in (self.Hterms if self.Htl>0 else [self.Hterms]):
            static1 = []
            for Ht in Ham:
                static1.append([Ht.upper(), [[-1*Ham[Ht]*Ham.sitefun(j)]+[(j+k)%self.L for k in range(len(Ht))] for j in range(self.L) if (self.pbc or j+len(Ht)<=self.L) and (Ham.sitefun(j)!=0)]])
                static1.append([Ht.lower(), [[+1*Ham[Ht]*Ham.sitefun(j)]+[(j+k)%self.L for k in range(len(Ht))] for j in range(self.L) if (self.pbc or j+len(Ht)<=self.L) and (Ham.sitefun(j)!=0)]])
            fprint(f"Hamiltonian part: {static1}")
            Hs.append(hamiltonian(static1, [], basis=basis, dtype=np.complex128, check_symm=False, check_pcon=False, check_herm=False))

        minLval = 1e10

        static2 = []
        for Lt in self.Lterms:
            if 0 < (ltabs:=np.abs(self.Lterms[Lt])) < minLval:
                minLval = ltabs
            static2.append([Lt, [[1j*self.Lterms[Lt]*self.Lterms.sitefun(j)]+[(j+k)%self.L for k in range(len(Lt))] for j in range(self.L) if (self.pbc or j+len(Lt)<=self.L) and self.Lterms.sitefun(j)!=0]])
            if Lt == "C":
                static2.append(["c", [[1j*self.Lterms[Lt]*self.Lterms.sitefun(j), j] for j in range(self.L) if (self.pbc or j+len(Lt)<=self.L) and self.Lterms.sitefun(j)!=0]])

        self.invert_sigma = minLval

        fprint(f"Dissipation part: {static2}")
        fprint(f"Preconditioning sigma set to {minLval}")

        H2 = hamiltonian(static2, [], basis=basis, dtype=np.complex128, check_symm=False, check_pcon=False, check_herm=False)

        if self.FlqT == 0:

            if self.sparse:
                self.L0mat = Hs[0].tocsr()
                for H in Hs[1:]:
                    self.L0mat += H.tocsr()
                self.L0mat *= -1j
                self.Ldmat = -1j*H2.tocsr()
            else:
                self.L0mat = Hs[0].toarray()
                for H in Hs[1:]:
                    self.L0mat += H.toarray()
                self.L0mat *= -1j
                self.Ldmat = -1j*H2.toarray()
            self.Lmat = self.L0mat + self.Ldmat

        else:

            Htl = max(self.Htl, 1)

            if self.sparse:
                
                Hs = [H.tocsr() for H in Hs]
                H2 = H2.tocsr()

                def evolve (v):
                    v1 = np.ascontiguousarray(v)
                    for H in Hs:
                        expm_multiply_parallel(H,a=-1j*self.FlqT,dtype=complex).dot(v1, overwrite_v=True)
                    expm_multiply_parallel(H2,a=-1j*self.FlqT,dtype=complex).dot(v1, overwrite_v=True)
                    return v1
                
                self.Lmat = LinearOperator(shape=np.shape(H2), matvec=evolve, dtype=complex)

            else:

                ### Use dense UF

                # Instantiate the Floquet system
                F = Floquet({"H_list": Hs + [H2],
                              "dt_list": [self.FlqT]*(Htl+1)}, UF=True)
                self.Lmat = F.UF


        span_bare = np.arange(4**self.L)[::-1]
        spanop_bare = np.zeros(np.shape(span_bare))
        for _ in range(self.L):
            spanop_bare += (span_bare%4 != 0)
            span_bare = span_bare // 4
        del span_bare
        spanop_bare = sparse.dia_matrix((spanop_bare, 0), shape=(4**self.L, 4**self.L))

        if self.sparse:
            self.spanop = spanop_bare
        else:
            projer = basis.get_proj(dtype=complex)
            self.spanop = projer.transpose().conj() @ spanop_bare @ projer
        fprint(f"Spanop constructed: {hasattr(self, 'spanop')}")
        
        self.basis = basis
        self.Ns = basis.Ns
        
        fprint(f"Lindblad construction cost {time()-t1}s")

        self.hasmat = True
        
    def mat (self, attr=-1):
        if not hasattr(self, "Lmat"):
            self.construct_mat()
        if attr < 0:
            return self.Lmat
        elif attr == 0:
            return self.L0mat
        else:
            return self.Ldmat

    def __repr__ (self):
        str = "Lindbladian(\n"
        str += "\nHamiltonian = \n"
        str += self.H.__repr__() + '\n'
        str += "\nJump Operators = \n"
        str += self.Ls.__repr__() + '\n'
        str += "\nConserved Quantities = \n"
        for iom in self.iom:
            str += iom.__repr__() + '\n'
        str += ")\n"
        return str
    

    def vs_to_ops (self, v, nterms = 0, rcut = 0, normalize = False):

        Lmap = {0:"I", 1:"X", 2:"Y", 3:"Z"}

        eigvecs = []

        for i in range(np.shape(v)[1]):

            vec = v[:,i] / np.linalg.norm(v[:,i])
            index_order = np.argsort(-np.abs(vec))

            this_ev = []
            leading_term = vec[index_order[0]]

            for j in range(len(index_order)):

                coef = vec[index_order[j]]

                if (nterms > 0 and j >= nterms) or (np.abs(coef/leading_term) < rcut):
                    break

                if normalize:
                    coef /= leading_term

                s = self.basis[index_order[j]]
                termn = ""
                length = self.L

                for _ in range(length):
                    termn += Lmap[s%4]
                    s = s//4

                if self.k is not None:

                    norm_factor = np.sqrt(length / cyclic_order(termn,length))

                    No_i = 1
                    while termn.find("I"*No_i) >= 0:
                        No_i += 1
                    No_i -= 1

                    translation = 0
                    if No_i > 0:
                        ind = termn.find("I"*No_i)
                        termn = termn[ind:] + termn[:ind]
                        translation = ind

                    termn = termn.strip("I")
                    if termn == "":
                        termn = "I"

                    phase = self.k*translation/self.L if self.k is not None else 0

                    this_ev.append((coef * np.exp(2j*np.pi*phase) * norm_factor, termn))

                else:

                    this_ev.append((coef, termn))

            op_dict = {}
            for coeff, term in this_ev:
                if term not in op_dict:
                    op_dict[term] = 0
                op_dict[term] += coeff
            eigvecs.append(Operator(op_dict))

        return eigvecs

    
    def getSpectrum (self, savename = "", override_existing_file = False, sparseno = 0, minconverge=0, lind = 20, nterms=10, rcut=0, get_svd=False, allow_order=2, cut_ioms=True, invert_sigma = None, get_overlaps = True, output=fprint):

        self.construct_mat()

        if self.Ns <= 5000 and sparseno > 0:
            fprint("Small matrix encountered, using dense diagonalization.")
            sparseno = 0

        if savename == "":
            savename = self.name

        if sparseno > 0:
            if savename.find("sp") == -1:
                savename += f"sp{sparseno}"
            self.sparse = True
            if minconverge <= 0:
                minconverge = sparseno
        else:
            self.sparse = False

        if invert_sigma is None and hasattr(self, "invert_sigma"):
            invert_sigma = self.invert_sigma

        try:
            if override_existing_file:
                raise Exception("OVERRIDING")
            
            data = np.load(npz_path(savename))
            w = data["w"]
            v = data["v"]
            output(f"Read data from {npz_path(savename)}")
            if self.sparse and len(w) < minconverge:
                raise Exception("NOT ENOUGH EIGENVALUES READ")

        except Exception as e:

            fprint(f"Encountered ::{e}:: while reading, re-doing diagonalization.")

            t1 = time()

            if sparseno > 0:
                ncv0 = min(self.Ns, max(2*sparseno + 1, 20))
                ncv = min(self.Ns, ncv0)
                while True:
                    output(f"Diagonalizing with ncv = {ncv} :: sparseno = {sparseno}, Ns = {self.Ns}.")

                    # We will generate a random initial vector that lies much in the few-operator subspace
                    # To do this, we first initialize a fully random vector
                    v0 = np.random.randn(self.Ns, 2)
                    v0 = v0[:,0] + 1j*v0[:,1]
                    # Then we scale this by a profile exp(-const*span)
                    # Here we choose const = 2
                    cexp = 2
                    scaler = np.exp(-cexp * self.spanop.diagonal())
                    v0 = np.abs(self.basis.project_to(scaler.astype(complex), sparse=False, pcon=False)).flatten() * v0
                    # Finally, normalize v0
                    v0 /= np.linalg.norm(v0)

                    try:
                        if self.FlqT == 0:
                            if invert_sigma is None:
                                output(f"Running sparse diagonalization with LR, no sigma.")
                                w, v = sparseeig(self.mat(), sparseno, which="LR", ncv=ncv, v0=v0)
                            else:
                                output(f"Running sparse diagonalization with preconditioned sigma = {invert_sigma}.")
                                w, v = sparseeig(self.mat(), sparseno, which="LM", sigma=invert_sigma, ncv=ncv, v0=v0)
                        else:
                            output(f"Running sparse diagonalization with LM, no sigma.")
                            w, v = sparseeig(self.mat(), sparseno, which="LM", ncv=ncv, v0=v0)
                        if np.max(np.real(w) if self.FlqT==0 else np.log(np.abs(w))) > 0.1:
                            output("Converges to wrong eigenvalues:", w, "restarting.")
                        break
                    except ArpackNoConvergence as e:
                        w = e.eigenvalues
                        output(f"Sparse diagonalization met with ArpackNoConvergence. Converged eigenvalue count: {len(w)}")
                        if len(w) < minconverge:
                            if ncv == self.Ns:
                                raise Exception("Already tried maximal ncv, exiting with no return.")
                            else:
                                ncv = min(self.Ns, ncv+ncv0)
                                continue
                        v = e.eigenvectors
                        break
            else:
                w, v = np.linalg.eig(self.mat())
            output(f"Lindblad diagonalization cost {time()-t1}s")

        results = {}

        if self.FlqT == 0:
            xs = np.real(w)
            ys = np.imag(w)
        else:
            xs = np.log(np.abs(w)) / self.FlqT
            ys = np.angle(w)

        warg = np.argsort(-xs)
        w = w[warg]
        v = v[:,warg]
        xs = xs[warg]
        ys = ys[warg]

        Neigs = len(w)

        results["xs"] = xs
        results["ys"] = ys
        results["w"] = w
        results["v"] = v

        output("w:", w)

        np.savez(npz_path(savename), w=w, v=v)

        if self.sparse:
            vproj = self.basis.project_from(v, sparse=False, pcon=False)
            spans = np.array([np.real(vproj[:,i].conj().T @ self.spanop @ vproj[:,i]) for i in range(Neigs)])
        else: 
            spans = np.real(np.diag(v.conj().T @ self.spanop @ v))

        results["spans"] = spans

        nvecs = min(lind, Neigs)
        results["nvecs"] = nvecs
        eigvecs = self.vs_to_ops(v[:,:nvecs], nterms=nterms, normalize=True, rcut=rcut)
        results["eigvecs"] = eigvecs
        output("EIGVECS:")
        for iev, ev in enumerate(eigvecs):
            output(f"X_{iev+1} = ")
            output(ev)
        results["has_svd"] = False

        if get_svd:

            try:
                v1 = np.linalg.inv(v)
                norms = np.linalg.norm(v, axis=0) * np.linalg.norm(v1, axis=1)
                results["condn"] = norms
            except Exception as e:
                output("Getting svd failed:")
                output(e)
            results["has_svd"] = True

        ##### TRYING TO CONSTRUCT OVERLAPS WITH IOMS #####
        if get_overlaps:

            results["has_iom"] = False

            if len(self.iom) > 0:

                try:

                    nameid = 1
                    ioms = []

                    for i,iom in enumerate(self.iom):
                        if iom.name is None:
                            ioms.append((iom, f"I.o.M. {nameid}"))
                            nameid += 1
                        else:
                            ioms.append((iom, "$"+iom.name+"$"))

                    # Considering multiples of operators
                    for order in range(2, allow_order+1):
                        for indices in itertools.combinations_with_replacement(range(len(self.iom)), order):
                            product = product_op(self.iom[indices[0]], self.iom[indices[1]], self.L)
                            for i in indices[2:]:
                                product = product_op(product, self.iom[i], self.L)

                            nametokens = []
                            lastname = None
                            lastind = -1
                            power = 1

                            for i in (list(indices) + [-2]): # The final element is added to flush the stack
                                if i != lastind:
                                    if lastname != None:
                                        nametokens.append(lastname + (f"^{power}" if power>1 else ""))
                                    lastind = i
                                    if i == -2:
                                        break
                                    lastname = self.iom[i].name
                                    power = 1
                                else:
                                    power += 1
                            ioms.append((product, "$" + " \\cdot ".join(nametokens) + "$"))

                    ioms = [(Operator({"I":1}, "I"), "$I$")] + ioms

                    names = []

                    output("Integrals of motion: ")
                    for iom, name in ioms:
                        output(name, " :: ", iom)
                        names.append(name)

                    results["ioms"] = ioms

                    # We do a Gram-Schmidt, or QR decomposition, to orthonormalized the integrals of motion
                    # We also wish to reorder the integrals of motion such that they are ranked by the aggregate overlap
                    #  they have with the eigenstates

                    # First construct the matrix of all ioms
                    Liom = len(ioms)
                    #output("Number of ioms:")
                    #output(Liom)
                    names = []
                    M0 = np.zeros((self.basis.Ns, Liom), dtype=complex)
                    for i_iom in range(Liom):
                        iomvec = ioms[i_iom][0].tovec(self.basis, self.k)
                        M0[:,i_iom] = iomvec#/np.linalg.norm(iomvec)
                        output(f"iom {ioms[i_iom][1]} has norm {np.linalg.norm(iomvec)}")
                        names.append(ioms[i_iom][1])

                    overlaps = np.zeros((Liom, nvecs), dtype=float)

                    for i in range(nvecs):

                        vr = v[:,i]
                        vrCT = vr.conj().reshape((1,-1))

                        # For each column in v, we iteratively find out the iom with largest overlap against it.
                        Nproced = 0

                        # We do a linear transformation of M, in which we move columns with largest overlap against vr to the front
                        #  and orthogonalize the other columns against it
                        M = np.copy(M0)

                        # We also document the relative positions of the ioms before and after permutation
                        # pos[i] indicates the original position of the iom that is now in the i-th column of M
                        pos = np.arange(Liom)

                        zeros_count = 0

                        for j in range(Liom):

                            # Orthogonalize all the ioms with respect to the previous IoM
                            if j > 0:
                                overlap = (M[:,j-1]).conj().reshape((1,-1)) @ M[:,j:Liom-zeros_count]
                                M[:,j:Liom-zeros_count] -= M[:,j-1].reshape((-1,1)) @ overlap

                            output("j =", j, "zerocount", zeros_count)
                            
                            # Find the norms of the rest of the ioms and normalize
                            Mnorms = np.linalg.norm(M[:,j:Liom-zeros_count], axis=0)
                            output("Mnorms", Mnorms)
                            # For those whose are very small after orthogonalization, don't do the division
                            if cut_ioms:
                                zeropos = (Mnorms < 1e-10)
                            else:
                                zeropos = (Mnorms < 0)
                            Mnorms[zeropos] = 1
                            output("zeropos", zeropos)
                            # Normalized M
                            M[:,j:Liom-zeros_count] /= Mnorms
                            # Rank the ioms by the residual norm
                            normsort = np.argsort(-Mnorms)
                            permuter = np.arange(Liom)
                            permuter[j:Liom-zeros_count] = np.arange(j,Liom-zeros_count)[normsort]
                            M = M[:,permuter]
                            pos = pos[permuter]
                            output("pos", pos)
                            # Increase zeros_count
                            zeros_count += np.sum(zeropos)

                            # Find the M that has largest overlap with vr
                            v_overlaps = np.abs(vrCT @ M[:,j:]).flatten()
                            output("Overlaps: ", v_overlaps)
                            # Find the one with maximal overlap
                            maxpos = np.argmax(v_overlaps)
                            # If the maximal overlap < 1/4, end the process
                            if v_overlaps[maxpos] < 0: # 1/np.sqrt(10):
                                break
                            # Put the one with maximal overlap to position i
                            permuter = np.arange(Liom)
                            permuter[j] = j+maxpos
                            permuter[j+maxpos] = j
                            M = M[:,permuter]
                            pos = pos[permuter]
                            output("Permuter:", permuter)

                            if not np.allclose(M[:,:j+1].conj().T@M[:,:j+1], np.eye(j+1)):
                                output(f"Q NOT UNITARY!")

                            Nproced += 1

                        if Nproced != 0:

                            v_overlaps = np.zeros((Liom,))
                            v_overlaps[:Nproced] = np.abs(vrCT @ M[:,:Nproced])**2

                            output("v_overlaps for ", i, "is", v_overlaps, "sum", np.sum(v_overlaps))

                            #output("v_overlaps", v_overlaps)
                            #output("pos", pos, "i", i)

                            overlaps[pos,i] = v_overlaps

                    #output("overlaps", overlaps)
                    max_overlap = np.max(overlaps, axis=1)
                    select_ioms = np.argsort(-max_overlap)
                    if cut_ioms:
                        select_ioms = select_ioms[max_overlap[select_ioms]>0.1]
                    #output("max_overlap", max_overlap)
                    #output("select_ioms", select_ioms)
                    overlaps = overlaps[select_ioms, :]
                    names = [names[i] for i in select_ioms]
                    #output("overlaps", overlaps)
                    #output("names", names)

                    #vnorms = np.linalg.norm(vr, axis=0)
                    vrest = 1 - np.sum(overlaps, axis=0)# / vnorms

                    results["overlaps"] = overlaps
                    results["vrest"] = vrest
                    results["names"] = names
                    results["full_names"] = names
                    #results["full_names"] = full_names

                    results["has_iom"] = True

                except Exception as e:
                    output("Getting iom failed:")
                    raise e

        ##### END TRYING TO CONSTRUCT OVERLAPS WITH IOMS #####

        return results
    
    def plotSpectrum (self, savename = "", compare = None, plot_svd = False, override_existing_file = False, lind = 20, allow_order=2, sparseno = 0):

        if savename == "":
            savename = self.name

        if sparseno > 0:
            if savename.find("sp") == -1:
                savename += f"sp{sparseno}"
            self.sparse = True
        else:
            self.sparse = False

        results = self.getSpectrum(savename=savename, override_existing_file=override_existing_file,
                                   sparseno=sparseno, lind=lind, get_svd=plot_svd, allow_order=allow_order)

        if results["has_iom"] and results["has_svd"]:
            fig, ((ax1, ax2), (axsvd, axiom)) = plt.subplots(2, 2)
            fig.set_figwidth(12)
            fig.set_figheight(11)
        elif results["has_iom"]:
            fig, (ax1, ax2, axiom, axlegend) = plt.subplots(1, 4, width_ratios=[2,2,2,1])
            axlegend.axis("off")
            fig.set_figwidth(16)
            fig.set_figheight(5)
        elif results["has_svd"]:
            fig, (ax1, ax2, axsvd) = plt.subplots(1, 3)
            fig.set_figwidth(16)
            fig.set_figheight(5)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_figwidth(13)
            fig.set_figheight(5)

        xs = results["xs"]
        ys = results["ys"]
        w = results["w"]

        spans = results["spans"]
        eigvecs = results["eigvecs"]
        nvecs = results["nvecs"]
        
        hasiom = results["has_iom"]
        if hasiom:
            overlaps = results["overlaps"]
            vrest = results["vrest"]
            names = results["names"]
            full_names = results["full_names"]
            nioms = len(full_names)
            if nioms == 0:
                hasiom = False

        f = open(savename + ".txt", 'w+')

        if hasiom:
            f.write("Normalized integrals of motion:\n\n")
            for i, fn in enumerate(full_names):
                f.write(f"Q{i+1} = {fn}\n")
            f.write("\n")

        for i in range(nvecs):

            f.write(f"Eigenvalue {w[i]}, span {spans[i]}\n")
            for coef,oprt in eigvecs[i]:
                f.write("{:.3f} {}\n".format(coef, oprt))
            if hasiom:
                f.write("\n")
                f.write("Overlaps:\n")
                for j in range(len(full_names)):
                    f.write(f"Q{j+1} x {overlaps[j,i]}\n")
                f.write("Residual {:.3f}\n".format(vrest[i]))
            f.write("\n")

        f.close()

        ax1.scatter(xs, ys, c=-spans)

        norm_factor = 1 if self.FlqT==0 else self.FlqT
        if np.max(np.abs(ys)) < 0.08*norm_factor:
            ax1.set_ylim([-0.1*norm_factor, 0.1*norm_factor])
        
        if self.FlqT == 0:
            xlabel = r"$\mathrm{Re} \lambda$"
        else:
            xlabel = r"$\mathrm{Re} \lambda / T$"
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"$\mathrm{Im} \lambda$")
        ax1.set_title("Lindblad Spectrum")

        if compare is not None:

            alpha = 2 * np.sqrt(np.trace(self.Hmat**2) / (self.sz * 2**self.sz))
            beta, gamma = compare
            smallgamma = beta*gamma < alpha/(2**self.sz)

            fprint(alpha, beta, gamma)
        
            ratio = alpha*np.sqrt(self.sz)
            cys = np.linspace(-2*ratio, 2*ratio)
            x0 = -beta*self.sz
            xdif = 2*np.pi*(beta*gamma)**2*self.sz/ratio*np.array([G(y/ratio) for y in ys])
            
            if smallgamma:
                xdif *= alpha/(2*(2**self.sz)*beta*gamma)

            linetype = "dashed" if smallgamma else "solid"

            ax1.plot(x0-xdif, cys, c="C1", linestyle=linetype)
            ax1.plot(x0+xdif, cys, c="C1", linestyle=linetype)

        ax2.scatter(xs, spans, c=-spans)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Operator Span")

        if results["has_svd"]:

            axsvd.scatter(-xs, results["condn"])
            axsvd.plot(-xs, -xs)
            axsvd.set_xlabel("|Re(lambda)|")
            axsvd.set_ylabel("Condition number")

        if hasiom:

            ploti = np.arange(nvecs)+1
            ploty = [[] for _ in range(len(names))]

            for i in range(nvecs):
                for j in range(nioms):
                    ploty[j].append(np.abs(overlaps[j,i]))

            axim = axiom.twiny()
            axiom,axim = axim,axiom

            axim.plot(xs[:nvecs], ploti, color="gray", alpha=0.7)
            axim.set_xlabel(xlabel)

            markers = ["o", "^", "s", "*", "v"]

            for j in range(nioms):
                if j < 100:
                    axiom.scatter(ploty[j], ploti, label=names[j], color=f"C{j}" if len(names)<=10 else plt.get_cmap('tab20')(j%20), marker=markers[j//20])
                else:
                    axiom.scatter(ploty[j], ploti, marker="x")

            axiom.scatter(vrest[:nvecs], ploti, label="Other", c="black", marker="x")
            axiom.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1+(len(names)//15))
            axiom.set_xlabel("Eigenvector overlap with I.o.M.")
            axiom.set_ylabel("# Eigenvector")

        if savename != "":
            fig.savefig(savename + ".pdf")
        
        fig.show()

    def plotSpectrum_Pie (self, savename = "", override_existing_file = False, lind = 20, allow_order=2, sparseno = 0):

        if savename == "":
            savename = self.name

        if sparseno > 0:
            if savename.find("sp") == -1:
                savename += f"sp{sparseno}"
            self.sparse = True
        else:
            self.sparse = False

        results = self.getSpectrum(savename=savename, override_existing_file=override_existing_file,
                                   sparseno=sparseno, lind=lind, get_svd=False, allow_order=allow_order,
                                   nterms=0, rcut=0.05)

        width = 2
        plt.rcParams.update(RCPARAMS)
        
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, width_ratios=[5,1], height_ratios=[2,1], figsize=(width*6/5, width*6/5))
        # plt.rcParams.update({'font.size': 8})
        ax2.axis("off")
        ax3.axis("off")
        ax4.axis("off")

        xs = results["xs"]
        ys = results["ys"]
        w = results["w"]

        spans = results["spans"]
        eigvecs = results["eigvecs"]
        nvecs = results["nvecs"]
        
        hasiom = results["has_iom"]
        if hasiom:
            overlaps = results["overlaps"]
            vrest = results["vrest"]
            names = results["names"]
            full_names = results["full_names"]
            nioms = len(full_names)
            if nioms == 0:
                hasiom = False


        f = open(savename + "Pie.txt", 'w+')

        if hasiom:
            f.write("Normalized integrals of motion:\n\n")
            for i, fn in enumerate(full_names):
                f.write(f"Q{i+1} = {fn}\n")
            f.write("\n")

        for i in range(nvecs):

            f.write(f"X{i+1} = \n")
            f.write(f"Eigenvalue {w[i]}, span {spans[i]}\n")
            # print("eigvecs[i] type: ", type(eigvecs[i]))
            # print("eigvecs[i]: ", eigvecs[i])
            for oprt in eigvecs[i]:
                coef = eigvecs[i][oprt]
                f.write("{:.3f} {}\n".format(coef, oprt))
            if hasiom:
                f.write("\n")
                f.write("Overlaps:\n")
                for j in range(len(full_names)):
                    f.write(f"Q{j+1} x {overlaps[j,i]}\n")
                f.write("Residual {:.3f}\n".format(vrest[i]))
            f.write("\n")

        f.close()

        sct = ax1.scatter(xs, ys, c=-spans)
        norm_factor = 1 if self.FlqT==0 else self.FlqT
        if np.max(np.abs(ys)) < 0.08*norm_factor:
            ax1.set_ylim([-0.1*norm_factor, 0.1*norm_factor])
        if self.FlqT == 0:
            xlabel = r"$\mathrm{Re} \lambda$"
        else:
            xlabel = r"$\mathrm{Re} \lambda / T$"
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"$\mathrm{Im} \lambda$")
        # ax1.set_title("Lindbladian Spectrum")

        plt.colorbar(sct, ax=ax2, label="Op. Size")

        def getcolor (n):
            return colormaps["tab20"](((3*n)%20)/20)

        def getpie (overlaps):
            N = len(overlaps)
            overlaps = np.array(overlaps)
            ranking = np.argsort(-overlaps)
            pie_data = overlaps[ranking]
            pie_data = np.append(pie_data, max(0, 1-np.sum(pie_data)))
            colors = [getcolor(i) for i in ranking] + ["white"]
            return pie_data, colors
        
        if nvecs <= 10:
            xno = 5
            yno = 2
        else:
            xno = 10
            yno = 4
        
        for i in range(nvecs):
            if i >= xno*yno:
                continue
            x = i % xno
            y = i // xno
            inset_ax = inset_axes(ax3, width="100%", height="100%", loc='center', bbox_to_anchor=(x/xno, (1-(y+1)/yno), 1/xno, 1/yno), 
                      bbox_transform=ax3.transAxes)
            pie_data, colors = getpie(overlaps[:,i])
            inset_ax.pie(pie_data, colors=colors, startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})

        legend_handle = [Patch(facecolor=getcolor(i), edgecolor='black', label=full_names[i]) for i in range(nioms)]
        ax4.legend(handles=legend_handle)

        if savename != "":
            fig.savefig(savename + "Pie.pdf")
        
        fig.show()


    def plotSpectrum_bar (self, results = None, savename = "", override_existing_file = False, lind = 20, allow_order=2, sparseno = 0, width = 3, iomcut=0.1, eigcut=0.1, output = print):

        if savename == "":
            savename = self.name

        if sparseno > 0:
            if savename.find("sp") == -1:
                savename += f"sp{sparseno}"
            self.sparse = True
        else:
            self.sparse = False

        if results is None:
            results = self.getSpectrum(savename=savename, override_existing_file=override_existing_file, get_overlaps=False,
                                    sparseno=sparseno, lind=lind, get_svd=False, allow_order=allow_order)

        eigvecs = results["v"]
        nvecs = results["nvecs"]

        lind = min(lind, nvecs)

        # Construct all integrals of motions as products of the original ioms

        ioms = []
        
        for i,iom in enumerate(self.iom):
            if iom.name is None:
                ioms.append((iom, f"I.o.M. {nameid}"))
                nameid += 1
            else:
                ioms.append((iom, "$"+iom.name+"$"))

        # Considering multiples of operators
        for order in range(2, allow_order+1):
            for indices in itertools.product(*([range(len(self.iom))]*order)):
                product = product_op(self.iom[indices[0]], self.iom[indices[1]], self.L)
                for i in indices[2:]:
                    product = product_op(product, self.iom[i], self.L)

                nametokens = []
                lastname = None
                lastind = -1
                power = 1

                for i in (list(indices) + [-2]): # The final element is added to flush the stack
                    if i != lastind:
                        if lastname != None:
                            nametokens.append(lastname + (f"^{power}" if power>1 else ""))
                        lastind = i
                        if i == -2:
                            break
                        lastname = self.iom[i].name
                        power = 1
                    else:
                        power += 1
                ioms.append((product, "$" + " \\cdot ".join(nametokens) + "$"))

        ioms = [(Operator({"I":1}, "I"), "$I$")] + ioms

        output("Integrals of motion: ")
        for iom, name in ioms:
            output(name)

        # For each IoM, we will find the overlap with the eigenstates
        # First we orthogonalize the eigenstates
        vr = eigvecs[:,:lind]
        ws = results["w"][:lind]
        print("Eigenvalues:")
        print(ws)
        select = np.abs(np.imag(ws)) < 1
        print("Select:")
        print(select)
        vladder = vr[:,np.logical_not(select)]
        wladder = ws[np.logical_not(select)]
        vr = vr[:,select]
        ws = ws[select]
        lind = len(ws)
        Qvr,_ = np.linalg.qr(vr)

        # Turn all ioms into vectors
        Liom = len(ioms)
        output("Number of ioms:")
        output(Liom)
        eff_iom = 0
        M0 = np.zeros((self.basis.Ns, Liom), dtype=complex)
        Mnets = np.zeros((self.basis.Ns, Liom), dtype=complex)
        names = []
        for i_iom in range(Liom):
            iomvec = ioms[i_iom][0].tovec(self.basis, self.k)
            iomvec /= np.linalg.norm(iomvec)
            iomvecP = np.copy(iomvec)
            for j in range(eff_iom):
                iomvecP -= Mnets[:,j] * np.inner(np.conj(Mnets[:,j]), iomvecP)
            P_overlap = np.linalg.norm(Qvr.conj().T @ iomvecP)

            if eff_iom > 0 and P_overlap < np.sqrt(0.1):
            #np.linalg.norm(M0[:,:eff_iom] @ np.linalg.lstsq(M0[:,:eff_iom], iomvec)[0] - iomvec) < 1e-6:
                output(f"iom {ioms[i_iom][1]} is a linear combination of the existing ones (subspace reduced)")
                continue
            else:
                # output("Added in: ", ioms[i_iom][0])
                output(f"iom {ioms[i_iom][1]} included with P_overlap = {P_overlap}")
                M0[:,eff_iom] = iomvec # / np.linalg.norm(iomvec)
                Mnets[:,eff_iom] = iomvecP / np.linalg.norm(iomvecP)
                names.append(ioms[i_iom][1])
                eff_iom += 1
        Liom = eff_iom
        output("Number of ioms after linear combination cut:", Liom)
        M0 = M0[:,:Liom]

        # Get column sums
        col_sums = np.sum(np.abs(Qvr.conj().T @ M0)**2, axis=0)
        print("Col_sums:")
        print(list(zip(names, col_sums)))
        # Get relevant ioms
        iomcut = 0.3
        iom_sel = np.arange(Liom)[col_sums > iomcut]
        col_sums = col_sums[iom_sel]
        M0 = M0[:,iom_sel]
        names = [names[i] for i in iom_sel]
        Liom = len(col_sums)
        output("Number of ioms after cut:")
        output(Liom)

        # Get the row sums
        QM,_ = np.linalg.qr(M0)
        row_sums = np.sum(np.abs(QM.conj().T @ vr)**2, axis=0)

        # Test IOM and vr
        if hasattr(self, "L0mat"):
            print("IOM and vr test:")
            print(np.apply_along_axis(np.linalg.norm, 0, self.L0mat@QM))
            print(np.apply_along_axis(np.linalg.norm, 0, self.L0mat@Qvr))
        
        overlaps = np.abs(vr.conj().T @ M0)**2

        m = lind # Number of rows
        n = Liom # Number of columns

        # Define figure and grid layout
        sum_panel = min(m,n)/4
        bar_panel = min(m,n)/8
        margin_panel = min(m,n)/4

        width = 3
        plt.rcParams.update(RCPARAMS)

        fig = plt.figure(figsize=(width, width*(m+sum_panel+margin_panel)/(n+sum_panel+bar_panel+2*margin_panel)))
        gs = fig.add_gridspec(3, 5, width_ratios=[margin_panel, n, sum_panel, bar_panel, margin_panel], height_ratios=[sum_panel, m, margin_panel], wspace=0.05, hspace=0.05)

        # Right plot (row sums)
        ax_bottom = fig.add_subplot(gs[1, 2])
        ax_bottom.barh(np.arange(m)+1, row_sums, color="green", alpha=0.6)
        ax_bottom.set_ylim((0.5, m+0.5))
        ax_bottom.axvline(1, color="black", linestyle="--")
        ax_bottom.set_ylabel(r"$R_i$", rotation=0, position=(-1,-0.15), ha="center", va="top")
        # ax_bottom.set_title("Integral of Motion Overlap Sum", rotation=-90, position=(1, 0.5), ha='left', va='bottom')

        # Top plot (column sums)
        ax_left = fig.add_subplot(gs[0, 1])
        ax_left.bar(np.arange(n)+1, col_sums, color="blue", alpha=0.6)
        ax_left.axhline(1, color="black", linestyle="--")
        ax_left.set_xlim((0.5, n+0.5))
        ax_left.set_ylabel(r"$C_\alpha$")

        # Remove extra ticks for a clean look
        ax_left.set_xticks([])
        ax_bottom.set_yticks([])

        # Main heatmap
        ax_main = fig.add_subplot(gs[1, 1])


        # Define a scaling function (e.g., power-law transformation)
        def scaling_function(x, gamma=0.5):  # gamma < 1 brightens, gamma > 1 darkens
            return x**gamma

        # Get an existing colormap (e.g., "viridis") and its colormap data
        base_cmap = plt.get_cmap("inferno")

        # Apply the scaling function to modify the colormap
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            "scaled_viridis",
            base_cmap(scaling_function(np.linspace(0, 1, 256)))  # Apply transformation
        )

        img = ax_main.imshow(overlaps[::-1,:], extent=[0.5, n+0.5, 0.5, m+0.5], cmap=new_cmap)
        ax_main.set_xticks(np.arange(n)+1)
        # Assign custom labels
        col_labels = names
        ax_main.set_xticklabels(col_labels, rotation=45, ha="right")
        ax_main.set_yticks(np.arange(m)+1)
        ax_main.set_yticklabels([f"$V_{{{i+1}}}$" for i in range(m)])
        # Ensure ticks match grid spacing
        ax_main.tick_params(axis="both", which="both", length=0)  # Hide tick marks

        bar = fig.colorbar(img, cax=fig.add_subplot(gs[1, 3]))
        bar.set_label(r"$A_{\alpha i}$")

        fig.canvas.draw()
        if savename != "":
            fig.savefig(savename + "Bar.pdf")
            fig.savefig(savename + "Bar.eps")
            output("Figure saved.")
        plt.close(fig)

        ### SAVE EIGVECS

        # Frist project the eigenvectors against QM and get the rest
        QvM, _ = np.linalg.qr(np.hstack((QM, vr)))
        # We have (QM, vr) = (QM, QvM2) (I RvM1; 0 RvM2)
        # That is, vr = QM*RvM1 + QvM2*RvM2.
        # Now we further diagonalize QvM2.
        sz1 = np.shape(QM)[1]
        QvM2 = QvM[:,sz1:]

        # We diagonalize QvM2 on the span operator
        if self.sparse:
            vproj = self.basis.project_from(QvM2, sparse=False, pcon=False)
            spans_op = vproj.conj().T @ self.spanop @ vproj
        else: 
            spans_op = np.real(np.diag(QvM2.conj().T @ self.spanop @ QvM2))
        
        sps, spv = np.linalg.eigh(spans_op)
        # We have sps[i] = span of the i-th eigenvector of QvM2
        # The ultimate vectors are given by QvM2 @ spv
        # We also have QvM2 = QvM1*RvM1 + QvM2*RvM2
        Xvs = QvM2 @ spv
        perp_ops = self.vs_to_ops(Xvs, rcut=0.05, normalize=True)

        f = open(savename + "Bar.txt", 'w+')

        f.write("Integrals of motion:\n\n")
        for i, fn in enumerate(names):
            f.write(f"Q{i+1} = {fn}\n")
        f.write("\n")

        f.write("Eigenvalues:\n")
        f.write(", ".join(map(str, ws)))
        f.write("\n\n")

        f.write("Overlaps:\n\n")
        f.write("row_sums = \n")
        f.write(", ".join(map(str, row_sums)))
        f.write("\ncol_sums = \n")
        f.write(", ".join(map(str, col_sums)))
        f.write("\noverlaps = \n")
        for i in range(overlaps.shape[0]):
            f.write("[" + ", ".join(map(str, overlaps[i,:])) + "], \n")
        f.write("\n\n")

        f.write("Leftover eigenvectors:\n\n")
        for i in range(len(sps)):
            f.write(f"# {i+1}\n")
            f.write(f"span = {sps[i]}\n\n")
            if hasattr(self, "L0mat"):
                vec = perp_ops[i].tovec(self.basis, self.k, self.pbc)
                if len(perp_ops[i].terms) < 20:
                    pop1 = self.vs_to_ops(vec[:,np.newaxis], rcut=0.05, normalize=True)
                    print(f"POP{i+1}:")
                    print(str(perp_ops[i]))
                    print(str(pop1))
                f.write(f"Commutator with H = {np.linalg.norm(self.L0mat @ vec)/np.linalg.norm(vec)}\n\n")
            f.write(str(perp_ops[i]) + "\n\n")
            f.write(str(perp_ops[i].terms))
            f.write("\n\n\n\n")

        if len(wladder) > 0:
            f.write("Ladder operators:\n")
            lad_ops = self.vs_to_ops(vladder, rcut=0.05, normalize=True)
            for i in range(len(wladder)):
                f.write(f"# {i+1}\n")
                f.write(f"Eigenvalue = {wladder[i]}\n\n")
                if hasattr(self, "L0mat"):
                    vec = vladder[:,i]
                    f.write(f"Commutator with H = {np.linalg.norm(self.L0mat @ vec)/np.linalg.norm(vec)}\n\n")
                f.write(str(lad_ops[i]) + "\n\n")
                f.write(str(lad_ops[i].terms))
                f.write("\n\n\n\n")
        else:
            f.write("No ladder operators.\n\n")
        
        f.close()