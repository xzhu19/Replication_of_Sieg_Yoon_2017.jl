# Replication code for Sieg and Yoon (2017)

module HelperFunctions 

using Plots
using NLopt
using Random
using LinearAlgebra
using Dierckx, GridInterpolations
using QuadGK
using DataFrames
using StatsFuns

function v_diff_d(x)
    vd = Spline1D(x_grid, vvd; k=3)[x] # Cubic spline
    vr = Spline1D(x_grid, vvr; k=3)[x]
    v = -abs(xt_1-x) + lambda*at_1 + beta*(vd*p_d + vr*(1-p_d))-ecost_d

    f = v - vr

    return f
end

function v_diff_r(x)
    vd = Spline1D(x_grid, vvd; k=3)[x]
    vr = Spline1D(x_grid, vvr; k=3)[x]
    v = -abs(xt_1-x) + lambda*at_1 + beta*(vd*p_d + vr*(1-p_d))-ecost_r

    f = v - vd

    return f
end

function v_snp_r(x)
    vd = Spline1D(x_grid, vvd; k=1)(x) # Linear spline (interpolation)
    vr = Spline1D(x_grid, vvr; k=1)(x)
    
    f = -abs(xt_1-x) + lambda*at_1 -ecost_r+beta*(vd*p_d + vr*(1-p_d))

    return f
end    

end