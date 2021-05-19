module fun_myfun_ntl2
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export myfun_ntl2

function myfun_ntl2(x)
    #= # Calculate the system of equations in no term limit case
    global beta lambda ecost_d ecost_r y_r y_d
    global n_app pdf_d pdf_r a_grid
    global u_d2 l_d2 u_r2 l_r2
    global fxr fxd =#
    
    n = n_app
    
    v_d = x[1]
    v_r = x[2]
    
    u_d = zeros(n,1)
    l_d = zeros(n,1)
    u_r = zeros(n,1)
    l_r = zeros(n,1)
    
    v_d_grid = zeros(1,n)
    v_r_grid = zeros(1,n)
    
    p1r = zeros(n,1)
    p2r = zeros(n,1)
    p3r = zeros(n,1)
    p5r = zeros(n,1)
    p6r = zeros(n,1)
    
    g1r = zeros(n,1)
    g2r = zeros(n,1)
    g3r = zeros(n,1)
    g5r = zeros(n,1)
    g6r = zeros(n,1)
    
    p1d = zeros(n,1)
    p2d = zeros(n,1)
    p3d = zeros(n,1)
    p5d = zeros(n,1)
    p6d = zeros(n,1)
    
    g1d = zeros(n,1)
    g2d = zeros(n,1)
    g3d = zeros(n,1)
    g5d = zeros(n,1)
    g6d = zeros(n,1)
    
    for i in 1:n
        if (lambda*a_grid[i]-ecost_r)/(1-beta)- v_d >=0 
            u_r[i] =  (lambda*a_grid[i]-ecost_r)-v_d*(1-beta)
            l_r[i] = - u_r[i]
        else
            u_r[i] = -0.01
            l_r[i] =  0.01
        end
    
        if (lambda*a_grid[i]-ecost_d)/(1-beta)-v_r >=0
            u_d[i] =  (lambda*a_grid[i]-ecost_d)-v_r*(1-beta)
            l_d[i] = - u_d[i]
        else
            u_d[i] = -0.01
            l_d[i] =  0.01   
        end
    end

    fun1 = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
    fun2 = y -> abs(y).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
    fun3 = y -> (fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
    fun4 = y -> abs(y).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
    
    p_all_r = integral(fun2,-inf,inf)
    p_all_d = integral(fun4,-inf,inf)
    
    for i in 1:n
        if (u_r[i] >= l_r[i])    
            p1r[i] = integral(fun1,-inf,l_r[i]-y_r[i])
            p2r[i] = integral(fun1,l_r[i]-y_r[i],l_r[i])
            p5r[i] = integral(fun1,u_r[i],u_r[i]+y_r[i])
            p6r[i] = integral(fun1,u_r[i]+y_r[i],inf)
            p3r[i] = max(0,1 - p1r[i] - p2r[i] - p5r[i] - p6r[i])
            g1r[i] = integral(fun2,-inf,l_r[i]-y_r[i])
            g2r[i] = integral(fun2,l_r[i]-y_r[i],l_r[i])
            g5r[i] = integral(fun2,u_r[i],u_r[i]+y_r[i])
            g6r[i] = integral(fun2,u_r[i]+y_r[i],inf)
            g3r[i] = max(0,p_all_r - g1r[i] - g2r[i] - g5r[i] - g6r[i])
        end
    
        if u_d[i] >= l_d[i] 
            p1d[i] = integral(fun3,-inf,l_d[i]-y_d[i])
            p2d[i] = integral(fun3,l_d[i]-y_d[i],l_d[i])
            p5d[i] = integral(fun3,u_d[i],u_d[i]+y_d[i])
            p6d[i] = integral(fun3,u_d[i]+y_d[i],inf)
            p3d[i] = max(0,1 - p1d[i] - p2d[i] - p5d[i] - p6d[i])
            
            g1d[i] = integral(fun4,-inf,l_d[i]-y_d[i])
            g2d[i] = integral(fun4,l_d[i]-y_d[i],l_d[i])
            g5d[i] = integral(fun4,u_d[i],u_d[i]+y_d[i])
            g6d[i] = integral(fun4,u_d[i]+y_d[i],inf)
            g3d[i] = max(0,p_all_d - g1d[i] - g2d[i] - g5d[i] - g6d[i])
        end
    end
    
    for i in 1:n
        if u_r[i] >= l_r[i]
            v_r_grid[i] = (p6r[i])*(lambda*a_grid[i]+beta*v_d)-g6r[i] +(p5r[i])*(-abs(u_r[i]) + lambda*a_grid[i]-beta*ecost_r)/(1-beta)+(p3r[i])*(lambda*a_grid[i]-beta*ecost_r)/(1-beta)-g3r[i]/(1-beta)+(p2r[i])*(-abs(l_r[i]) + lambda*a_grid[i]-beta*ecost_r)/(1-beta) +(p1r[i])*(lambda*a_grid[i]+beta*v_d)-g1r[i]
        else
            v_r_grid[i] = -p_all_r+lambda*a_grid[i]+beta*v_d
        end

        if u_d[i] >= l_d[i]
            v_d_grid[i] = (p6d[i])*(lambda*a_grid[i]+beta*v_r)-g6d[i] +(p5d[i])*(-abs(u_d[i]) + lambda*a_grid[i]-beta*ecost_d)/(1-beta) +(p3d[i])*(lambda*a_grid[i]-beta*ecost_d)/(1-beta)-g3d[i]/(1-beta) +(p2d[i])*(-abs(l_d[i]) + lambda*a_grid[i]-beta*ecost_d)/(1-beta) +(p1d[i])*(lambda*a_grid[i]+beta*v_r)-g1d[i]
        else
            v_d_grid[i] = -p_all_d+lambda*a_grid[i]+beta*v_r
        end
    end

    f[1] = sum(v_d_grid.*pdf_d) - v_d
    f[2] = sum(v_r_grid.*pdf_r) - v_r
    
    l_d2 = l_d
    l_r2 = l_r
    u_d2 = u_d
    u_r2 = u_r

    return f
end