__author__ = 'Xinyue'
from linearize import linearize
from linearize import linearize_para

def convexify_para_obj(obj):
    '''
    input:
        obj: an objective of a problem
    return:
        if the objective is dcp,
        return the cost function (an expression);
        if the objective has a wrong curvature,
        return the linearized expression of the cost function,
        the zeros order parameter,
        the dictionary of parameters indexed by variables,
        the domain
    '''
    if obj.is_dcp() == False:
        return linearize_para(obj.args[0])
    else:
        return obj.args[0]

def is_dccp(objective):
    """
    input:
        objective: an objective of a problem
    return:
        if the objective is dccp
        the objective must be convex, concave, affine, or constant
    """
    if objective.args[0].curvature == 'UNKNOWN':
        return False
    else:
        return True

# the following function is not used anymore in the parameterized version
def convexify_obj(obj):
    if obj.is_dcp() == False:
        result = linearize(obj.args[0])
    else:
        result = obj.args[0]
    return result