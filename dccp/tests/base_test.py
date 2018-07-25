# Base class for unit tests.
import unittest
import numpy as np


class BaseTest(unittest.TestCase):
    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places=5):
        a = self.mat_to_list(a)
        b = self.mat_to_list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=5):
        super(BaseTest, self).assertAlmostEqual(a,b,places=places)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat

    #Test function to check if computed solution, comp_sol, is approximately equal to any of the possible solutions, sols.
    def assertIsAlmostIn(self, comp_sol, sols, tolerance=0.000001):
        '''
        Input: comp_sol - the computed solution in the optimization problem
               sols - list of all possible solutions
               tolerance - tolerance that they are almost equal
        '''
        comp_sol = self.mat_to_list(comp_sol)
        sols = self.mat_to_list(sols)
        truth = [np.linalg.norm(np.asarray(comp_sol) - np.asarray(sol_ex)) < tolerance for sol_ex in sols]
        self.assertTrue(any(truth))