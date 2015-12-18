import unittest
import sys
sys.path.insert(1, "../src")
import numpy as np
from chemistry import split_species, split_species_reaction, calc_molecular_weight
from chemistry import Chemistry

class TestUtils(unittest.TestCase):
    def test_split(self):
        s = "H2O"
        split_s = split_species(s)
        self.assertEqual(split_s[0], ("H", 2))
        self.assertEqual(split_s[1], ("O", 1))

        s = "H2O2"
        split_s = split_species(s)
        self.assertEqual(split_s[0], ("H", 2))
        self.assertEqual(split_s[1], ("O", 2))


        s = "H2O1L1N10M11Pa43"
        split_s = split_species(s)
        self.assertEqual(split_s[0], ("H", 2))
        self.assertEqual(split_s[1], ("O", 1))
        self.assertEqual(split_s[2], ("L", 1))
        self.assertEqual(split_s[3], ("N", 10))
        self.assertEqual(split_s[4], ("M", 11))
        self.assertEqual(split_s[5], ("Pa", 43))

    def test_split_species_reaction(self):
        s = "10*H3O"
        split = split_species_reaction(s)
        self.assertEqual(split[0], "H3O")
        self.assertEqual(split[1][0], ("H", 3))
        self.assertEqual(split[1][1], ("O", 1))
        self.assertEqual(split[2], 10)
        
    def test_calc_molecular_weight(self):
        ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
        s = split_species("H2O2")
        mw = calc_molecular_weight(s)
        self.assertAlmostEqual(mw, 34.0/1e3)
        
        s = split_species("H2O")
        mw = calc_molecular_weight(s)
        self.assertAlmostEqual(mw, 18.0/1e3)

        s = split_species("O2")
        mw = calc_molecular_weight(s)
        self.assertAlmostEqual(mw, 32.0/1e3)

        s = split_species("H2")
        mw = calc_molecular_weight(s)
        self.assertAlmostEqual(mw, 2.0/1e3)

        s = split_species("H2N2O2")
        mw = calc_molecular_weight(s)
        self.assertAlmostEqual(mw, 62.0/1e3)

    def test_Chemistry_1(self):
        ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
        species = ['H2', 'O2', 'H2O']
        enthalpy = [0.0, 0.0, 0.0]
        reaction = {'equation':'H2 + 0.5*O2 = H2O', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': 1.5e6}
        c = Chemistry(species, enthalpy)
        c.add_reaction(reaction)
        self.assertAlmostEqual(c.reactions[0].Q, 1.5e6)
        self.assertAlmostEqual(c.reactions[0].A, 1e9)
        self.assertAlmostEqual(c.reactions[0].b, 1.0)
        self.assertAlmostEqual(c.reactions[0].Ta, 14000.0)
        
        self.assertAlmostEqual(c.reactions[0].nurhs.tolist(), [0.0, 0.0, 1.0])
        self.assertAlmostEqual(c.reactions[0].nulhs.tolist(), [1.0, 0.5, 0.0])
        self.assertEqual(c.reactions[0].lhs_species, ["H2", "O2"])
        self.assertEqual(c.reactions[0].rhs_species, ["H2O"])

        Y = np.ones([3, 10])
        Y[0,:] = 0.2
        Y[1,:] = 0.2
        Y[2,:] = 0.6
        X = c.massf_to_molef(Y)
        sum_ybyw = 0.2/2.0 + 0.2/32.0 + 0.6/18.0
        self.assertAlmostEqual(X[0,0], 0.2/2.0/(sum_ybyw))
        self.assertAlmostEqual(X[1,0], 0.2/32.0/(sum_ybyw))
        self.assertAlmostEqual(X[2,0], 0.6/18.0/(sum_ybyw))


    def test_Chemistry_1(self):
        ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
        species = ['H2', 'O2', 'H2O', 'H2O2', 'O3', 'O']
        enthalpy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        reaction = {'equation':'2*H2 + O + 2.5*O2 = H2O + H2O2 + O3', 'A': 1e9, 'b': 1.0, 'Ta':14000.0, 'Q': 1.5e6}
        c = Chemistry(species, enthalpy)
        c.add_reaction(reaction)
        self.assertAlmostEqual(c.reactions[0].Q, 1.5e6)
        self.assertAlmostEqual(c.reactions[0].A, 1e9)
        self.assertAlmostEqual(c.reactions[0].b, 1.0)
        self.assertAlmostEqual(c.reactions[0].Ta, 14000.0)
        
        self.assertAlmostEqual(c.reactions[0].nurhs.tolist(), [0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(c.reactions[0].nulhs.tolist(), [2.0, 2.5, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(c.reactions[0].lhs_species, ["H2", "O", "O2"])
        self.assertEqual(c.reactions[0].rhs_species, ["H2O", "H2O2", "O3"])

        Y = np.ones([6, 10])
        Y[0,:] = 0.2
        Y[1,:] = 0.2
        Y[2,:] = 0.2
        Y[3,:] = 0.1
        Y[4,:] = 0.13
        Y[5,:] = 0.17
        X = c.massf_to_molef(Y)
        ybyw_H2 = 0.2/2.0
        ybyw_O2 = 0.2/32.0
        ybyw_H2O = 0.2/18.0
        ybyw_H2O2 = 0.1/34.0
        ybyw_O3 = 0.13/48.0
        ybyw_O = 0.17/16.0

        sum_ybyw = ybyw_H2 + ybyw_O2 + ybyw_H2O + ybyw_H2O2 + ybyw_O3 + ybyw_O
        self.assertAlmostEqual(X[0,0], ybyw_H2/(sum_ybyw))
        self.assertAlmostEqual(X[1,0], ybyw_O2/(sum_ybyw))
        self.assertAlmostEqual(X[2,0], ybyw_H2O/(sum_ybyw))
        self.assertAlmostEqual(X[3,0], ybyw_H2O2/(sum_ybyw))
        self.assertAlmostEqual(X[4,0], ybyw_O3/(sum_ybyw))
        self.assertAlmostEqual(X[5,0], ybyw_O/(sum_ybyw))
        
    def test_Chemistry_2(self):
        ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
        species = ['H2', 'O2', 'H2O', 'H2O2', 'O3', 'O']
        enthalpy = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        c = Chemistry(species, enthalpy)
        reaction = {'equation':'2*H2 + O + 2.5*O2 = H2O + H2O2 + O3', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
        c.add_reaction(reaction)
        reaction = {'equation':'H2 + O + 1.5*O2 = H2O + O3', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
        c.add_reaction(reaction)


        self.assertAlmostEqual(c.reactions[0].Q, 50.0 + 40.0 + 30.0 - 2*10.0 - 2.5*20.0 - 60.0)
        self.assertAlmostEqual(c.reactions[0].A, 1e9)
        self.assertAlmostEqual(c.reactions[0].b, 1.0)
        self.assertAlmostEqual(c.reactions[0].Ta, 14000.0)
        self.assertAlmostEqual(c.reactions[1].Q, 30.0 + 50.0 - 10.0 - 60.0 - 1.5*20.0)

        
        self.assertAlmostEqual(c.reactions[0].nurhs.tolist(), [0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(c.reactions[0].nulhs.tolist(), [2.0, 2.5, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(c.reactions[0].lhs_species, ["H2", "O", "O2"])
        self.assertEqual(c.reactions[0].rhs_species, ["H2O", "H2O2", "O3"])

        self.assertAlmostEqual(c.reactions[1].nurhs.tolist(), [0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(c.reactions[1].nulhs.tolist(), [1.0, 1.5, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(c.reactions[1].lhs_species, ["H2", "O", "O2"])
        self.assertEqual(c.reactions[1].rhs_species, ["H2O", "O3"])


    def test_Chemistry_3(self):
        ELEMENTS = {'O':16.0/1e3, 'N':14.0/1e3, 'H':1.0/1e3}
        species = ['H2', 'O2', 'H2O', 'H2O2', 'O3', 'O']
        enthalpy = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        c = Chemistry(species, enthalpy)
        reaction = {'equation':'2*H2 + O + 2.5*O2 = H2O + H2O2 + O3', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
        c.add_reaction(reaction)
        reaction = {'equation':'H2 + O + 1.5*O2 = H2O + O3', 'A': 1e9, 'b': 1.0, 'Ta':14000.0}
        c.add_reaction(reaction)
        
        T = np.zeros([10], dtype=np.complex) + 1e-16
        Y = np.ones([6, 10], dtype=np.complex)
        Y[0,:] = 0.2
        Y[1,:] = 0.2
        Y[2,:] = 0.2
        Y[3,:] = 0.1
        Y[4,:] = 0.13
        Y[5,:] = 0.17
        source_T, source_Y = c.calc_source_terms(T, Y)
        self.assertEqual(np.allclose(source_T, 0.0), True)
        self.assertEqual(np.allclose(source_Y, 0.0), True)

        T = np.ones([10], dtype=np.complex)*1000
        source_T, source_Y = c.calc_source_terms(T, Y)
        self.assertGreater(source_T.tolist(), 0.0)
        self.assertGreater(source_Y.tolist(), 0.0)
        self.assertEqual(np.unique(source_T).size, 1)
        for i in range(6):
            self.assertEqual(np.unique(source_Y[0,:]).size, 1)


if __name__ == "__main__":
    unittest.main()
