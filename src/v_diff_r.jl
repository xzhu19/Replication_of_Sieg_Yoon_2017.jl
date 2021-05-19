module fun_v_diff_r
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export v_diff_r

using GridInterpolation 
##Importing Global variables
import beta
import p_d
import xt_1 
import vvd
import vvr 
import x_grid 
import at_1 
import lambda 
import ecost_r


vr = interpolate(x_grid, vvr, x)
vd = interpolate(x_grid, vvd, x)

#v function 
v = -abs(xt_1 - x) + lambda*at_1 + beta * (vd *p_d + vr *(1-p_d)) -ecost_r

# v diff function 
function  v_diff_r(x) = v-vd

end