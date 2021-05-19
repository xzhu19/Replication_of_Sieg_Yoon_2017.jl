module fun_v_func_snp
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export v_func_snp

function v_func_snp(x)
    global y_d y_r beta p_d u_d l_d u_r l_r theta
    global n_app pdf_d pdf_r
    global lambda a_grid ecost_d ecost_r
    global fxr fxd

    v_d = x[1]
    v_r = x[2]
    v_0 = v_d*p_d + v_r*(1-p_d)

    f[1] = 0

    for k in 1:n_app
        if l_d[k] <= u_d[k]
        f1d = y -> (-abs(y-theta) +lambda*a_grid[k]+ beta*v_r).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
        f2d = y -> (-abs(l_d[k]-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid[k]-beta*ecost_d+ beta^2*v_0).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
        f3d = y -> (-(1+beta)*abs(y-theta) + (1+beta)*lambda*a_grid[k]-beta*ecost_d+ beta^2*v_0).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
        f4d = y -> (-abs(u_d[k]-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid[k]-beta*ecost_d+ beta^2*v_0).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
        f[1] = f[1] + (integral(f1d,-inf,l_d[k]-y_d[k]) + integral(f2d,l_d[k]-y_d[k],l_d[k]) ...
                    + integral(f3d,l_d[k],u_d[k]) + integral(f4d,u_d[k],u_d[k]+y_d[k]) ...
                    + integral(f1d,u_d[k]+y_d[k],inf))*pdf_d[k]
                
        else
        f1d = y -> (-abs(y-theta) +lambda*a_grid[k] + beta*v_r).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
        f[1] = f[1] + integral(f1d,-inf,inf)*pdf_d[k]
        end       
            
    end

    f[1] = f[1] - v_d

    f[2] = 0

    for k in 1:n_app
        if l_r[k] <= u_r[k]
            f1r = y -> (-abs(y-theta) +lambda*a_grid[k]+ beta*v_d).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
            f2r = y -> (-abs(l_r[k]-theta) -beta*abs(y-theta) + (1+beta)*lambda*a_grid[k]-beta*ecost_r+ beta^2*v_0).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
            f3r = y -> (-(1+beta)*abs(y-theta) + (1+beta)*lambda*a_grid[k]-beta*ecost_r+ beta^2*v_0).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
            f4r = y -> (-abs(u_r[k]-theta) -beta*abs(y-theta) + (1+beta)*lambda*a_grid[k]-beta*ecost_r+ beta^2*v_0).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
            f(2) = f(2) + (integral(f1r,-inf,l_r[k]-y_r[k]) + integral(f2r,l_r[k]-y_r[k],l_r[k]) ...
                        + integral(f3r,l_r[k],u_r[k]) + integral(f4r,u_r[k],u_r[k]+y_r[k]) ...
                        + integral(f1r,u_r[k]+y_r[k],inf))*pdf_r[k]   
        else
        f1r = y -> (-abs(y-theta) +lambda*a_grid[k]+ beta*v_d).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
        f[2] = f[2] + integral(f1r,-inf,inf)*pdf_r[k]     
        end  
    end

    f[2] = f[2] - v_r
    return f
end

end