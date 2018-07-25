"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division

from dccp.tests.base_test import BaseTest
import cvxpy as cvx
from dccp.objective import convexify_obj
from dccp.constraint import convexify_constr
from dccp.linearize import linearize
import dccp.problem
import dccp
import numpy as np

class TestExample(BaseTest):
    """ Unit tests example. """
    def setUp(self):
        # Initialize things.
        self.a = cvx.Variable(1)
        self.x = cvx.Variable(2)
        self.y = cvx.Variable(2)
        self.z = cvx.Variable(2)

    def test_readme_example(self):
        """
        Test the example in the readme.
        self.sol - All known possible solutions to the problem in the readme.
        """
        self.sol = [[0,0], [0,1], [1,0], [1,1]]
        myprob = cvx.Problem(cvx.Maximize(cvx.norm(self.y-self.z,2)), [0<=self.y, self.y<=1, 0<=self.z, self.z<=1])
        assert not myprob.is_dcp()   # false
        assert dccp.is_dccp(myprob)  # true
        result = myprob.solve(method = 'dccp')
        #print(self.y.value, self.z.value)
        self.assertIsAlmostIn(self.y.value, self.sol)
        self.assertIsAlmostIn(self.z.value, self.sol)
        self.assertAlmostEqual(result[0], np.sqrt(2))

    def test_linearize(self):
        """
        Test the linearize function.
        """
        z = cvx.Variable((1,5))
        expr = cvx.square(z)
        z.value = np.reshape(np.array([1,2,3,4,5]), (1,5))
        lin = linearize(expr)
        self.assertEqual(lin.shape, (1,5))
        self.assertItemsAlmostEqual(lin.value, [1,4,9,16,25])

    def test_convexify_obj(self):
        """
        Test convexify objective
        """
        obj = cvx.Maximize(cvx.sum(cvx.square(self.x)))
        self.x.value = [1,1]
        obj_conv = convexify_obj(obj)
        prob_conv = cvx.Problem(obj_conv, [self.x <= -1])
        prob_conv.solve()
        self.assertAlmostEqual(prob_conv.value,-6)

        obj = cvx.Minimize(cvx.sqrt(self.a))
        self.a.value = [1]
        obj_conv = convexify_obj(obj)
        prob_conv = cvx.Problem(obj_conv,cvx.sqrt(self.a).domain)
        prob_conv.solve()
        self.assertAlmostEqual(prob_conv.value,0.5)

    def test_convexify_constr(self):
        """
        Test convexify constraint
        """
        constr = cvx.norm(self.x) >= 1
        self.x.value = [1,1]
        constr_conv = convexify_constr(constr)
        prob_conv = cvx.Problem(cvx.Minimize(cvx.norm(self.x)), [constr_conv[0]])
        prob_conv.solve()
        self.assertAlmostEqual(prob_conv.value,1)

        constr = cvx.sqrt(self.a) <= 1
        self.a.value = [1]
        constr_conv = convexify_constr(constr)
        prob_conv = cvx.Problem(cvx.Minimize(self.a), [constr_conv[0],constr_conv[1][0]])
        prob_conv.solve()
        self.assertAlmostEqual(self.a.value[0],0)

    def test_vector_constr(self):
        """
        Test DCCP with vector cosntraints.
        """
        prob = cvx.Problem(cvx.Minimize(self.x[0]), [self.x >= 0])
        # doesn't crash with solver params.
        result = prob.solve(method="dccp", verbose=True)
        self.assertAlmostEqual(result[0], 0)