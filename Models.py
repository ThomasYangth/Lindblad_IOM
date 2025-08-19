from LindbladS import *
from OPlib import *

def Ising (sz, Jz, hz, hx, diss, noise, NoIOM=3, k=None, FlqT = 0, pbc=True, only_energy=False):

    H1 = Operator({"ZZ":Jz, "Z":hz}, "Hz")
    H2 = Operator({"X":hx}, "Hx")
    H = Operator({"ZZ":Jz, "Z":hz, "X":hx}, "H")

    L = {}
    # Depolarizing Noise
    if 'P' in noise:
        L["L"] = diss
    # Dephasing Noise
    if 'H' in noise:
        L["H"] = diss
    # Decay Noise
    if 'C' in noise:
        L["C"] = diss
    Ls = Operator(L)

    ioms = [H]

    if not only_energy:
        for m in range(1, NoIOM+1):
            Am = Operator({"Y"+"X"*(m-1)+"Z":1, "Z"+"X"*(m-1)+"Y":-1}, f"A_{m}")
            if m >= 2:
                Bm = Operator({"Z"+"X"*m+"Z":Jz, "Z"+"X"*(m-1)+"Z":-hx, "Y"+"X"*(m-1)+"Y":-hx, "Y"+"X"*(m-2)+"Y":Jz}, f"B_{m}")
            elif m == 1:
                Bm = Operator({"ZXZ":Jz, "ZZ":-hx, "YY":-hx, "X":-Jz}, "B_1")

            ioms += [Am, Bm]

    if FlqT == 0:
        return Lindbladian(H, Ls, sz, k, pbc=pbc, conserved_quantities=ioms, name=f"Ising_J{Jz}hz{hz}hx{hx}_sz{sz}k{k}_{noise}{diss}")
    else:
        return Lindbladian([H1,H2], Ls, sz, k, pbc=pbc, conserved_quantities=ioms, name=f"Ising_J{Jz}hz{hz}hx{hx}_sz{sz}k{k}_{noise}{diss}_F{FlqT}", Hterms_len=2, Floquet_t=FlqT)


def Ising_WP (sz, diss, noise, k=None, FlqT = 0, pbc=True):

    Jz = 1
    hz = 0.4
    hx = 0.4

    H1 = Operator({"ZZ":Jz, "Z":hz}, "Hz")
    H2 = Operator({"X":hx}, "Hx")
    H = Operator({"ZZ":Jz, "Z":hz, "X":hx}, "H")

    L = {}
    # Depolarizing Noise
    if 'P' in noise:
        L["L"] = diss
    # Dephasing Noise
    if 'H' in noise:
        L["H"] = diss
    # Decay Noise
    if 'C' in noise:
        L["C"] = diss
    Ls = Operator(L)

    N_op = Operator({"ZZ":0.953, "X":0.2135, "ZXZ":0.1927, "ZXXZ":0.0616, "ZX":-0.0398, "XZ":-0.0398, "YY":-0.0243}, "N")
    ioms = [H, N_op]

    if FlqT == 0:
        return Lindbladian(H, Ls, sz, k, pbc=pbc, conserved_quantities=ioms, name=f"Ising_J{Jz}hz{hz}hx{hx}WP_sz{sz}k{k}_{noise}{diss}")
    else:
        return Lindbladian([H1,H2], Ls, sz, k, pbc=pbc, conserved_quantities=ioms, name=f"Ising_J{Jz}hz{hz}hx{hx}WP_sz{sz}k{k}_{noise}{diss}_F{FlqT}", Hterms_len=2, Floquet_t=FlqT)


def XYZ (sz, Jx, Jy, Jz, hz, diss, noise, Jnn=0, k=None, FlqT = 0, noaprx = True):

    Hx = Operator({"XX":Jx}, "Hx")
    Hy = Operator({"YY":Jy}, "Hy")

    Hterms = {"ZZ":Jz}
    HtermsFull = {"XX":Jx, "YY":Jy, "ZZ":Jz}
    if hz != 0:
        Hterms["Z"] = hz
        HtermsFull["Z"] = hz
    if Jnn != 0:
        HtermsFull["XIX"] = Jnn
        HtermsFull["YIY"] = Jnn
        HtermsFull["ZIZ"] = Jnn
    Hz = Operator(Hterms, "Hz")

    H = Operator(HtermsFull, "H")

    L = {}
    # Depolarizing Noise
    if 'P' in noise:
        L["L"] = diss
    # Dephasing Noise
    if 'H' in noise:
        L["H"] = diss
    # Decay Noise
    if 'C' in noise:
        L["C"] = diss

    Ls = Operator(L)

    ioms = [H]
    # ioms = []

    if Jx==Jy or (np.abs(Jx-Jy) < 0.5 and not noaprx):  # Use approximation if Jx and Jy are equal or close enough
        ioms.append(Operator({"Z":1}, "S_z"))
    if Jy == Jz:
        ioms.append(Operator({"X":1}, "S_x"))
    if Jz == Jx:
        ioms.append(Operator({"Y":1}, "S_y"))
    # if np.abs(Jy-Jz) < 0.5 and np.abs(hz) < 0.5:
    #     ioms.append(Operator({"X":1}, "S_x"))
    # if np.abs(Jz-Jx) < 0.5 and np.abs(hz) < 0.5:
        # ioms.append(Operator({"Y":1}, "S_y"))
    if (hz == 0) or ((np.abs(hz) < 0.5) and not noaprx):
        ioms.append(Operator({"ZXY":1/Jx, "XYZ":1/Jy, "YZX":1/Jz, "YXZ":-1/Jx, "ZYX":-1/Jy, "XZY":-1/Jz}, "K"))
    # if max(np.abs([Jx-Jy, Jy-Jz, Jz-Jx])) < 0.5:
    #     ioms.append(Operator({"XYZ":1, "YZX":1, "ZXY":1, "YXZ":-1, "ZYX":-1, "XZY":-1}, "Q_2"))
    # elif np.abs(Jx-Jy) < 0.5:
    #     Delta = 2*Jz/(Jx+Jy)
    #     ioms.append(Operator({"XYZ":1, "YZX":Delta, "ZXY":1, "YXZ":-1, "ZYX":-1, "XZY":-Delta}, "Q_2"))
    #     ioms.append(Operator({"XYZ":1, "YZX":2-Delta, "ZXY":1, "YXZ":-1, "ZYX":-1, "XZY":-(2-Delta)}, "Q_3"))

    if not noaprx:
        Delta = np.abs(2*Jz/(Jx+Jy))
        if Delta < 1/4:
            ioms.append(Operator({"XY":1, "YX":-1}, "J"))
        elif Delta > 4:
            ioms.append(Operator({"XX":1, "YY":1, "ZXXZ":1, "ZYYZ":1}, "H_{\\text{D.W.}}"))
            # ioms.append(Operator({"XZY":1, "YZX":-1}, r"J_\text{D.W.}"))
        elif Delta != 1:
            ssq_dict = {}
            for i in range((sz-1) // 2):
                ssq_dict["X"+"I"*i+"X"] = 1
                ssq_dict["Y"+"I"*i+"Y"] = 1
            if sz % 2 == 0:
                ssq_dict["X"+"I"*(sz/2-1)+"X"] = 1/2
                ssq_dict["Y"+"I"*(sz/2-1)+"Y"] = 1/2

            ioms.append(Operator(ssq_dict, "(\\mathbf{S}^2)"))

    if FlqT == 0:
        return Lindbladian(H, Ls, sz, k, conserved_quantities=ioms, name=f"Hsb_X{Jx}Y{Jy}Z{Jz}h{hz}nn{Jnn}_sz{sz}k{k}_{noise}{diss}")
    else:
        return Lindbladian([Hx,Hy,Hz], Ls, sz, k, conserved_quantities=ioms, name=f"Hsb_X{Jx}Y{Jy}Z{Jz}h{hz}nn{Jnn}_sz{sz}k{k}_{noise}{diss}_F{FlqT}", Hterms_len=3, Floquet_t=FlqT)