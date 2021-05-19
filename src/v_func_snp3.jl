module fun_v_func_snp3

function v_func_snp3(x)
    global beta p_d u_d l_d u_r l_r theta u2_d l2_d u2_r l2_r
    global n_app pdf_d pdf_r
    global lambda a_grid ecost_d ecost_r 
    global fxr fxd
    
    v_d = x(1)
    v_r = x(2)
    v_0 = v_d*p_d + v_r*(1-p_d)
    
    f(1) = 0
    
    for k in 1:n_app
        if l_d(k) <= u_d(k)    
            f1d = y -> ((y< l2_d(k)|y>u2_d(k)).*(-abs(y-theta)+lambda*a_grid(k)+beta*v_r)...
                    +(y>=l2_d(k)&y< l_d(k)).*(-abs(l_d(k)-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid(k)-beta*ecost_d+ beta^2*v_0)...
                    +(y>=l_d(k)&y<= u_d(k)).*(-(1+beta)*abs(y-theta) + (1+beta)*lambda*a_grid(k)-beta*ecost_d+ beta^2*v_0)...
                    +(y> u_d(k)&y<=u2_d(k)).*(-abs(u_d(k)-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid(k)-beta*ecost_d+ beta^2*v_0)).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
            f(1) = f(1) + integral(f1d,-inf,inf)*pdf_d(k)     
        else
            f1d = y -> (-abs(y-theta) +lambda*a_grid(k) + beta*v_r).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
            f(1) = f(1) + integral(f1d,-inf,inf)*pdf_d(k)
        end       
    end
     
    f(2) = 0
    
    for k in 1:n_app
        if l_r(k) <= u_r(k)    
            f1r = y -> ((y< l2_r(k)|y>u2_r(k)).*(-abs(y-theta)+lambda*a_grid(k)+beta*v_r)...
                    +(y>=l2_r(k)&y< l_r(k)).*(-abs(l_r(k)-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid(k)-beta*ecost_r+ beta^2*v_0)...
                    +(y>=l_r(k)&y<= u_r(k)).*(-(1+beta)*abs(y-theta) + (1+beta)*lambda*a_grid(k)-beta*ecost_r+ beta^2*v_0)...
                    +(y> u_r(k)&y<=u2_r(k)).*(-abs(u_r(k)-theta) -beta*abs(y-theta) +(1+beta)*lambda*a_grid(k)-beta*ecost_r+ beta^2*v_0)).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2)
            f[2] = f[2] + integral(f1r,-inf,inf)*pdf_r(k)
        else
            f1r = y -> (-abs(y-theta) +lambda*a_grid(k)+ beta*v_d).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2)
            f[2] = f[2] + integral(f1r,-inf,inf)*pdf_r(k)   
        end   
    end
    
    return f
end