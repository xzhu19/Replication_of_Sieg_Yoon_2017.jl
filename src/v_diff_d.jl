module fun_v_diff_d
include("./src/Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export v_diff_d

global beta p_d  xt_1 vvd vvr x_grid at_1 lambda ecost_d

using GridInterpolations

function v_diff_d(x)
    vd = Spline1D(x_grid, vvd; k=3)[x] # Cubic spline
    vr = Spline1D(x_grid, vvr; k=3)[x]
    v = -abs(xt_1-x) + lambda*at_1 + beta*(vd*p_d + vr*(1-p_d))-ecost_d

    f = v - vr

    return f
end

end