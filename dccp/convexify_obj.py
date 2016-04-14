__author__ = 'Xinyue'
from linearize import linearize
from linearize import linearize_para

def convexify_obj(obj):
        if obj.is_dcp() == False:
            result = linearize(obj.args[0])
        else:
            result = obj.args[0]
        return result

def convexify_para_obj(obj):
        if obj.is_dcp() == False:
            result = linearize_para(obj.args[0])
        else:
            result = obj.args[0]
        return result

def is_dccp(objective):
    """The objective must be convex, concave, affine, or constant
    """
    if objective.args[0].curvature == 'UNKNOWN':
        return False
    else:
        return True
