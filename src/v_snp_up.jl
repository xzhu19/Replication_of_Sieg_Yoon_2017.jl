module fun_v_snp_up
include("./src/Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export v_snp_up

using QuadGK

function v_snp_up(x)
    global beta p_d vvd vvr x_grid a_i lambda a_grid ecost_r
    global fxr
    global u_r y_r

    vd = interp1(x_grid,vvd,x)
    vr = interp1(x_grid,vvr,x)

    fun1 = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
    fun2 = y -> y.*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)


    p5r = quadgk(fun1,u_r[a_i],u_r[a_i]+y_r[a_i])[1]
    g5r = quadgk(fun2,u_r[a_i],u_r[a_i]+y_r[a_i])[1]
    e5r = g5r/max(p5r,0.00001)

    p6r = quadgk(fun1,u_r[a_i],x)[1]
    g6r = quadgk(fun2,u_r[a_i],x)[1]
    e6r = g6r/max(p6r,0.00001)

    p7r = quadgk(fun1,x,u_r[a_i]+y_r[a_i])[1]
    g7r = quadgk(fun2,x,u_r[a_i]+y_r[a_i])[1]
    e7r = g7r/max(p7r,0.00001)

    if x<u_r[a_i] || x>u_r[a_i] + y_r[a_i]
        f = -abs(e5r-x) + lambda*a_grid[a_i] - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    else
        f = -abs(e6r-x)*p6r/(p6r+p7r) -abs(e7r-x)*p7r/(p6r+p7r)+ lambda*a_grid[a_i] - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    end

    return f
end

end