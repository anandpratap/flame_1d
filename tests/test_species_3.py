import sys
sys.path.insert(1, "../src")
from chemistry import Chemistry
from flame_1d import FlameBase
if __name__ == "__main__":
    # reaction takes place because the freestream source term 
    # do not cancel each other
    # All the Y_k becomes same after the flame, equals 0.67
    # temperature reaches ~ 2000K
    species = ['A', 'B', 'C', 'D']
    enthalpy = [0.0, 0.0, 1.5e6/2, 0.0]
    S_minf = [300.0, 0.1, 0.1, 0.0, 0.8]
    S_pinf = [900.0, 0.0, 0.0, 0.2, 0.8]

    c = Chemistry(species, enthalpy)
    reaction = {'equation':'A + B = 2*C', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
    c.add_reaction(reaction)    
    reaction = {'equation':'2*C = A + B', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
    c.add_reaction(reaction)
    flame = FlameBase(c, S_minf, S_pinf)
    flame.dt = 1e-3
    flame.solve()
