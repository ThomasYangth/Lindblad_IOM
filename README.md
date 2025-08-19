# Lindblad_IOM

This is the code that is used in the calculations of the paper Integrals of motion as slow modes in dissipative many-body operator dynamics ([https://arxiv.org/abs/2506.02970](arxiv:2506.02970)). The code is based on the open-source [https://quspin.github.io/QuSpin/index.html](Quspin library).

To use the code, first install QuSpin and relevant dependencies. Then, call `RunPlot.py` from the terminal. For example, to calculate the Lindbladian spectrum for a mixed-field Ising model with depolarizing noise of amplitude 0.1, use
```
python RunPlot.py python RunPlot.py --model=ising --hz=0.5 --hx=0.5 --nt=P --na=0.1 --L=8 --k=0
```
Most parameters are self-explanatory. More can be learned by looking at `RunPlot.py` and `Models.py`.

For any questions, please contact Tian-Hua Yang at [mailto:yangth@princeton.edu](yangth@princeton.edu).