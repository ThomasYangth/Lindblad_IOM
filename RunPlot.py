from LindbladS import *
from Models import *

from os import chdir
import sys

from scipy import sparse

# Use this to put all the data into a specific directory
# chdir("LindData")

args = {}

for arg in sys.argv:
    if arg.startswith("--") and len(arg) > 2:
        eqpos = arg.find("=")
        if eqpos < 0:
            args[arg[2:]] = True
        else:
            args[arg[2:eqpos]] = arg[eqpos+1:]
    else:
        print(f"Unrecognized command: '{arg}', skipped.")

print("Running with arguments:")
print(args)

# Get argument
def ga (argname, defval = 0., dtype=float, must=False):
    if argname in args:
        return dtype(args[argname])
    else:
        if must:
            print(f"Argument '{argname}' is required!")
            exit()
        print(f"Argument '{argname}' not provided, using {defval} as default.")
        return defval

if "model" not in args:
    print("'model' argument is required to run!")
    exit()
else:
    model = args["model"].lower()
    if model == "ising":
        Hammodel = Ising(ga("L",dtype=int,must=True), ga("J",1.), ga("hz"), ga("hx"), ga("na"), ga("nt","",str), k=ga("k",None,int), FlqT=ga("F",0,float), only_energy=ga("noaprx", False, bool))
    elif model == "isingwp":
        Hammodel = Ising_WP(ga("L",dtype=int,must=True), ga("na"), ga("nt","",str), k=ga("k",None,int), FlqT=ga("F",0,float))
    elif model == "hsb" or model == "heisenberg" or model == "xyz":
        Hammodel = XYZ(ga("L",dtype=int,must=True), ga("Jx",1.), ga("Jy",1.), ga("Jz",1.), ga("hz"), ga("na"), ga("nt","",str), Jnn=ga("Jnn"), k=ga("k",None,int), FlqT=ga("F",0,float), noaprx=ga("noaprx", False, bool))
    else:
        print(f"Unrecognized model name '{model}'.")
        exit()

    plottype = ga("plot", dtype=str, defval="pie").lower()
    if plottype == "pie":
        Hammodel.plotSpectrum_Pie(lind=ga("lind",10,int), allow_order=ga("iomorder",2,int), sparseno=ga("spn",0,int), override_existing_file=ga("override",False,bool))
    elif plottype == "bar":
        Hammodel.plotSpectrum_bar(lind=ga("lind",10,int), allow_order=ga("iomorder",2,int), sparseno=ga("spn",0,int), override_existing_file=ga("override",False,bool))
    else:
        print(f"Unrecognized plot type '{plottype}'.")
        exit()

#Ising (8, 1, 0, 2, 0.04, "P", k=0).plotSpectrum()
#XYZ (8, 1, 1, 1.5, 0, 0.04, "P", k=0).plotSpectrum()

#print(sparse.csr_matrix(Ising (3, 1, 0, 0.5, 0.04, "P", k=None).mat()[::-1,::-1]))
