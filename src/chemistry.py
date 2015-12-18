import numpy as np
import math
import re

ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
ELEMENTS['A'] = 0.029
ELEMENTS['B'] = 0.029
ELEMENTS['C'] = 0.029
ELEMENTS['D'] = 0.029

class Reaction(object):
    def __init__(self):
        pass

def split_species(s):
    composition = re.split('(\d+)',s)
    if composition[-1] == '':
        composition.pop()
    nspecies = int(math.ceil(len(composition)/2.0))
    species = []
    for i in range(nspecies):
        try:
            species.append((composition[2*i], int(composition[2*i+1])))
        except:
            species.append((composition[2*i], 1))
    return species

def split_species_reaction(s):
    s = s.replace(' ','')
    s = s.split("*")
    if len(s) == 1:
        s = s[0]
        fac = 1.0
    else:
        fac = float(s[0])
        s = s[1]
    composition = re.split('(\d+)',s)
    if composition[-1] == '':
        composition.pop()
    nspecies = int(math.ceil(len(composition)/2.0))
    species = []
    for i in range(nspecies):
        try:
            species.append((composition[2*i], int(composition[2*i+1])))
        except:
            species.append((composition[2*i], 1))
    return s, species, fac


def calc_molecular_weight(s):
    mw = 0.0
    for i in range(len(s)):
        mw += ELEMENTS[s[i][0]]*s[i][1]
    return mw

class Chemistry(object):
    def __init__(self, species, enthalpy):
        self.species = species
        self.enthalpy = enthalpy
        self.nspecies = len(species)
        self.setup()
        self.reactions = []

    def setup(self):
        self.obj_species = {}
        mw = self.calc_molecular_weight(self.species)
        self.mw = mw
        for s in range(len(self.species)):
            self.obj_species[self.species[s]] = mw[s]

    def add_reaction(self, reaction):
        r = self.check_reaction(reaction)
        self.reactions.append(r)
        return True

    def massf_to_molef(self, Y):
        n = np.shape(Y)[1]
        ybymw_sum = np.zeros(n, dtype=np.complex)
        for i in range(self.nspecies):
            ybymw_sum += Y[i,:]/self.mw[i]
        X = np.zeros_like(Y)
        for i in range(self.nspecies):
            X[i,:] = (Y[i,:]/self.mw[i])/ybymw_sum
        return X

    def calc_source_terms(self, T, Y):
        n = np.size(T)
        source_T = np.zeros(n, dtype=T.dtype)
        source_Y = np.zeros([self.nspecies, n], dtype=T.dtype)
        for idx, reaction in enumerate(self.reactions):
            Ta = reaction.Ta
            A = reaction.A
            b = reaction.b
            Q = reaction.Q
            Cp = 1005.0
            nu = reaction.nurhs - reaction.nulhs
            
            kf = A*T**b*np.exp(-Ta/T)
            mw = 0.029
            xprod = np.ones_like(kf)
            X = self.massf_to_molef(Y)
            for i in range(len(reaction.lhs_species)):
                p = reaction.nulhs[self.species.index(reaction.lhs_species[i])]
                xprod *= X[self.species.index(reaction.lhs_species[i]), :]**p
            q = kf*xprod
            w_dot = np.zeros([self.nspecies, n], dtype=np.complex)
            mw = self.mw
            for i in range(self.nspecies):
                w_dot[i,:] = nu[i]*q

            sum_w_dot = np.sum(w_dot, axis=0)
            
            w_dot_T = Q/Cp*q#(w_dot_p + w_dot_f + w_dot_o)
            #w_dot_T = Q/Cp*sum_w_dot
            source_T += w_dot_T
            for i in range(self.nspecies):
                source_Y[i,:] += w_dot[i,:]*mw[i]
        #ioff()
        #figure()
#        plot(source_T)
        #plot(source_Yo)
#        print nu
        #plot(source_Y.T)
        #show()
        
        return source_T, source_Y

    
    def check_reaction(self, reaction):
        nulhs = np.zeros(len(self.species))
        nurhs = np.zeros(len(self.species))
        equation = reaction['equation'].replace(" ","")
        lhs = equation.split("=")[0]
        rhs = equation.split("=")[1]
        lhs_s = []
        rhs_s = []

        mwlhs = 0.0
        for l in lhs.split("+"):
            s, this_species,fac = split_species_reaction(l)
            mwlhs += calc_molecular_weight(this_species)*fac
            nulhs[self.species.index(s)] += fac
            lhs_s.append(s)

        mwrhs = 0.0
        for l in rhs.split("+"):
            s, this_species, fac = split_species_reaction(l)
            mwrhs += calc_molecular_weight(this_species)*fac
            nurhs[self.species.index(s)] += fac
            rhs_s.append(s)

        assert(abs(mwrhs - mwlhs) < 1e-12)
        r = Reaction()
        r.lhs_species = lhs_s
        r.rhs_species = rhs_s
        r.nulhs = nulhs
        r.nurhs = nurhs
        
        try:
            Q = reaction['Q']
        except:
            Q = 0.0
            for i in range(len(r.rhs_species)):
                i_ = self.species.index(r.rhs_species[i])
                Q += self.enthalpy[i_]*r.nurhs[i_]
            for i in range(len(r.lhs_species)):
                i_ = self.species.index(r.lhs_species[i])
                Q -= self.enthalpy[i_]*r.nulhs[i_]
        r.A = reaction['A']
        r.b = reaction['b']
        r.Ta = reaction['Ta']
        r.Q = Q
        return r


    def calc_molecular_weight(self, species):
        mw = np.empty(len(species))
        for idx, s in enumerate(species):
            this_species = split_species(s)
            mw[idx] = calc_molecular_weight(this_species)
        return mw

if __name__ == "__main__":
    species = ['H2', 'O2', 'H2O']
    enthalpy = [0.0, 0.0, 0.0]
    reaction = {'equation':'H2 + 0.5*O2 = H2O', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': 1.5e6}
    c = Chemistry(species, enthalpy)
    c.add_reaction(reaction)
