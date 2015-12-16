import sys
sys.path.insert(1, "../src")
from chemistry import Chemistry
from flame_1d import FlameBase
if __name__ == "__main__":
    # source terms in the temperature cancel each other
    # no reactions, just transportation of species until everything
    # becomes constant
    species = ['A', 'B', 'C', 'D']
    enthalpy = [0.0, 0.0, 1.5e6/2, 0.0]
    S_minf = [300.0, 0.1, 0.1, 0.1, 0.7]
    S_pinf = [900.0, 0.0, 0.0, 0.0, 1.0]

    c = Chemistry(species, enthalpy)
    reaction = {'equation':'A + B = 2*C', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
    c.add_reaction(reaction)    
    reaction = {'equation':'2*C = A + B', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
    c.add_reaction(reaction)
    flame = FlameBase(c, S_minf, S_pinf)
    flame.dt = 1e-3
    flame.solve()
