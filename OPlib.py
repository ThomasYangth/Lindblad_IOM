# Operator library, updated 04/23/2024

import numpy as np

class Operator:

    def __init__ (self, termsmap, name=None, sitefun=lambda _:1):
        self.terms = termsmap
        self.name = name
        self.sitefun = sitefun

    def __iter__ (self):
        return iter(self.terms)
    
    def __getitem__ (self, key):
        return self.terms[key]
    
    def __repr__ (self):
        return " + ".join(["({:.3f}){}".format(self[k], k) for k in self])
    
    def __str__ (self):
        return self.__repr__()
    
    def tovec_full (self, L, trans_k = None, pbc=True):

        if not pbc:
            trans_k = None
    
        if trans_k is None:
            trans_k = 0

        v = np.zeros((4**L), dtype=complex)

        bases = np.array([4**(L-i-1) for i in range(L)])
        def toindex (arr):
            return np.sum(np.array(arr)*bases)
        ind = {"I":0, "X":1, "Y":2, "Z":3}

        for k in self:

            if len(k.upper().strip("I")) == 0:
                v[0] += self[k]*np.sum([self.sitefun(i) for i in range(L)])#*L
            
            else:
                for i in range(L if pbc else L+1-len(k)):
                   arr = [0]*L
                   for j,c in enumerate(k):
                       arr[(i+j)%L] = ind[c.upper()]
                   v[toindex(arr)] += self[k]*np.exp(-2j*np.pi*trans_k*i/L)*self.sitefun(i)

        return v[::-1]  # Return in reverse order to match the basis ordering
    
    def tovec (self, basis, trans_k = None, pbc=True):

        return basis.project_to(self.tovec_full(basis.N, trans_k, pbc), sparse=False, pcon=False)
    
IND_MAP = {"I":0, "X":1, "Y":2, "Z":3, 0:"I", 1:"X", 2:"Y", 3:"Z"}

# For now this works only for translationally invariant products 
def product_op (op1, op2, L, name = None):

    tmap = {}
    tm1 = op1.terms
    tm2 = op2.terms
    for k1 in tm1:
        s1 = k1 + "I"*(L-len(k1))
        for k2 in tm2:
            lk2 = len(k2)
            for i in range(L):
                s2 = k2[min(L-i, lk2):] + "I"*min(i, L-lk2) + k2[:min(L-i, lk2)] + "I"*max(0, L-i-lk2)
                c, s = str_times(s1, s2)

                c *= tm1[k1]*tm2[k2]

                # Find the best translation
                No_i = 1
                while s.find("I"*No_i) >= 0:
                    No_i += 1
                No_i -= 1
                if No_i > 0:
                    ind = s.find("I"*No_i)
                    s = s[ind+No_i:] + s[:ind+No_i]
                    s = s.strip("I")

                if s in tmap:
                    tmap[s] += c
                else:
                    tmap[s] = c
    return Operator(tmap, name)

# Initializations for multiplication functions
def single_times (id1, id2):
    if id1 == 0:
        return 1, id2 # Id times anything gives itself
    elif id2 == 0:
        return 1, id1 # Id times anything gives itself
    elif id1 == id2:
        return 1, 0 # Each of Sx, Sy, Sz squares to Id
    else:
        res = 6-id1-id2
        return (1j if id2-1==(res-2)%3 else -1j), res
    
def str_times (s1, s2):
    l = len(s1)
    if len(s2) != l:
        raise Exception(f"Can only multiply strings of equal lengths! Instead, got strings: '{s1}' and '{s2}'.")
    ss = ""
    coef = 1
    for i in range(l):
        c,s = single_times(IND_MAP[s1[i]], IND_MAP[s2[i]])
        coef *= c
        ss += IND_MAP[s]
    return coef, ss

# Multiply two terms
def term_times (inds1, inds2):
    coeff = 1
    if isinstance(inds1, Term):
        coeff *= inds1.coef
        inds1 = inds1.inds
    if isinstance(inds2, Term):
        coeff *= inds2.coef
        inds2 = inds2.inds
    l1 = len(inds1)
    l2 = len(inds2)
    if l1 != l2:
        raise Exception("Only terms with the same length can be multiplied!")
    ninds = np.zeros((l1,), dtype=int)
    for i in range(l1):
        c, r = single_times(inds1[i], inds2[i])
        coeff *= c
        ninds[i] = r
    return Term(coeff, ninds)

def to_square (arr):
    shape = np.shape(arr)
    dim = int(len(shape)/2)
    return arr.reshape((np.prod(shape[:dim]), np.prod(shape[dim:])))

def to_matrix_order (arr):
    dim = len(np.shape(arr))
    return arr.transpose([i for i in range(0, dim, 2)]+[i for i in range(1, dim ,2)])

def tensor_times (arr1, arr2):
    dim = int(len(np.shape(arr1))/2)
    if len(np.shape(arr2)) != dim*2:
        raise Exception("arr1 and arr2 must have the same and even dimensionality!")
    return np.tensordot(arr1, arr2, [[i for i in range(dim,2*dim)], [i for i in range(dim)]])

def extend_support (arr, old_support, new_support):

    if set(new_support) <= set(old_support):
        return arr

    paxes1 = []
    paxes2 = []
    p1 = 0 # Index of old array
    p2 = 0 # Index of appended zeros

    osl = len(np.shape(old_support))
    lsz = np.shape(arr)[0]

    for i,x in enumerate(new_support):
        if x in old_support:
            paxes1.append(p1)
            paxes2.append(osl+p1)
            p1 += 1
        else:
            arr = np.tensordot(arr, np.eye(lsz), axes=0)
            paxes1.append(2*osl+2*p2)
            paxes2.append(2*osl+2*p2+1)
            p2 += 1

    return np.transpose(arr, axes=paxes1+paxes2)

# The term class
class Term:

    def __init__ (self, coef, inds):
        self.coef = coef
        if isinstance(inds, str):
            self.inds = [IND_MAP[c] for c in inds]
        else:
            self.inds = inds

    def copy (self):
        return Term(self.coef, self.inds[:])

    def __imul__ (self, obj):
        print("imul called")
        if isinstance(obj, Term):
            t = self.__mul__(obj)
            self.coef = t.coef
            self.inds = t.inds
        else:
            self.coef *= obj
    
    def __mul__ (self, obj):
        if isinstance(obj, Term):
            return term_times(self, obj)
        else:
            return Term(self.coef*obj, self.inds)
    
    def __rmul__ (self, ratio):
        return self.__mul__(ratio)
    
    def conj (self):
        t1 = self.copy()
        t1.coef = np.conj(t1.coef)
        return t1

    def __repr__ (self):
        if self.coef == 0:
            return "0"
        elif len(self.inds) == 0:
            return "({:.2f})I".format(self.coef)
        else:
            return "({:.2f}){}".format(self.coef, "".join([IND_MAP[id] for id in self.inds]))

    def __str__ (self):
        return self.__repr__()
    
    def __iter__ (self):
        yield self.inds
        yield self.coef

    def __len__ (self):
        if self.coef == 0:
            return 0
        else:
            return len(self.inds)
    
    def getMats (self, type = "", add_coef = 1):
        if type not in ["","L","R"]:
            raise Exception("type must be either ''(Direct), 'L'eft, or 'R'ight!")
        if len(self) == 0:
            return []
        coef = self.coef * add_coef
        sgn = coef / np.abs(coef)
        val = np.abs(coef) ** (1/len(self))
        lst = [val*PAULIS[IND_MAP[ind]+type] for ind in self.inds]
        lst[0] *= sgn
        return lst
    
    def support (self):
        return [i for i in range(len(self)) if self.inds[i] != 0]
    
    def getMat (self, type = "", tensorshape = True, support = True):
        if type not in ["","L","R"]:
            raise Exception("type must be either ''(Direct), 'L'eft, 'R'ight!")
        if len(self) == 0:
            return 1
        m = self.coef
        for i,ind in enumerate(self.inds):
            if ind != 0 or support==True or (isinstance(support,list) and i in support):
                m = np.tensordot(m, PAULIS[IND_MAP[ind]+type], axes=0)
        
        m = to_matrix_order(m)
        if not tensorshape:
            m = to_square(m)
        return m
    
# Functions for multiplying operators on a lattice

# Expand a shorter index list to a longer one by adding zeros to it
def expand (arr, flen, pos = 0):
    l = len(arr)
    if pos < 0:
        return np.concatenate((np.zeros((flen-l,), dtype=int), arr))
    elif pos == 0:
        return np.concatenate((arr, np.zeros((flen-l,), dtype=int)))
    else:
        return np.concatenate((np.zeros((pos,), dtype=int), arr, np.zeros((flen-l-pos,), dtype=int)))

# Multiply two terms, taking results of a given length
def lat_times (inds1, inds2, flen):
    reslst = Operator()
    l1 = len(inds1)
    l2 = len(inds2)
    if flen > max(l1,l2):
        reslst.append(term_times(expand(inds1, flen), expand(inds2, flen, -1)))
        reslst.append(term_times(expand(inds1, flen, -1), expand(inds2, flen)))
    elif flen == l1:
        for i in range(flen-l2+1):
            reslst.append(term_times(inds1, expand(inds2, flen, i)))
    elif flen == l2:
        for i in range(flen-l1+1):
            reslst.append(term_times(expand(inds1, flen, i), inds2))
    else:
        raise Exception("flen must be no less than max(len(inds1),len(inds2))!")
    return reslst

# Multiply two terms and take all the results
def lat_times_all (inds1, inds2):
    reslst = Operator()
    l1 = len(inds1)
    l2 = len(inds2)
    for i in range(max(l1,l2), l1+l2):
        reslst.join(lat_times(inds1,inds2,i))
    return reslst