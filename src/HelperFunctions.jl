# Replication code for Sieg and Yoon (2017)

module HelperFunctions 

using Plots
using NLopt
using Random
using LinearAlgebra
using Dierckx
using QuadGK
using DataFrames

function chf1(t2, v1, v2)
    n = size(v1)[1]
    k1 = size(v1)[2]
    k2 = size(v2)[2]
    s2 = size(t2)

    if s2[1] > s2[2]
        t2 = transpose(t2)
        d = s2[1]
    else 
        d = s2[2]
    end

    w = 0
    v = 0
    dw = 0
    g1 = ones(1, d)
    g2 = ones(n, 1)
    l1 = 1

    while l1 <= k1
        l2 = 1
        while l2 <= k2
            w0 = sum(exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v2[:,l2])), dims=1)
            v0 = sum(exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v1[:,l1])), dims=1)
            dw0 = sum(sqrt(-1 + 0im)*kron(g1,v1[:,l1]).*exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v2[:,l2])), dims=1)
            dw = dw .+ dw0
            w = w .+ w0
            v = v .+ v0
            l2 = l2 + 1
        end
        l1 = l1 + 1
    end

    w = w/(n*k1*k2)
    v = v/(n*k1*k2)
    dw = dw/(n*k1*k2)

    if s2[1] > s2[2]
        w = transpose(w)
        v = transpose(v)
        dw = transpose(dw)
    end

    return [w, v, dw]
end

function cch2(x, v1, v2, p)
    (w, v, dw) = chf1(x, v1, v2)
    rr = dw./w
    if p == 0
        y = real(rr)
    else
        y = (rr - real(rr))/sqrt(-1 + 0im)
    end
    return y
end

function chfc2(v1, v2, T, m)
    tol = 0.01
    trace = 0
    t1 = (linspace(-T,T,m))'
    f1 = ones(m, 1)
    f2 = ones(m, 1)
    f3 = ones(m, 1)
    k = 1
    while k<=m
        p = 0
        c1 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        p = 1
        c2 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        f1(k,1) = exp(c1+1i*c2)
        (w,v,dw) = chf1(t1(k,1),v1,v2)
        f2(k,1) = v/f1(k,1)    
        f3(k,1) = w/f1(k,1)
        k = k + 1
    end

    return [f1, f2, f3]
end

function chfc3(v1, v2, v3, T, m)
    tol = 0.01
    trace = 0
    t1 = (linspace(-T,T,m))'
    f1 = ones(m, 1)
    f2 = ones(m, 1)
    f3 = ones(m, 1)
    k = 1
    while k<=m
        p = 0
        c1 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        p = 1
        c2 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        f1(k,1) = exp(c1+1i*c2)
        (w,v,dw) = chf1(t1(k,1),v1,v3)
        f2(k,1) = v/f1(k,1)    
        f3(k,1) = w/f1(k,1)
        k = k + 1
    end

    return [f1, f2, f3]
end

function mycon(x)
    x0 = x(1)
    x1 = x(2)
    x2 = x(3)
    x3 = x(4)
    x4 = x(5)
    mu = x(6)
    sig = x(7)
    
    c = []
    ceq = x0^2*sqrt(pi)*sig + (x1^2+2*x0*x2)*sqrt(pi)*sig^3/2 + (x2^2 + 2*x0*x4 + 2*x1*x3)*sqrt(pi)*sig^5*3/4 + (x3^2 + 2*x2*x4)*sqrt(pi)*sig^7*15/8 + (x4^2)*sqrt(pi)*sig^9*105/16 - 1

    return [c, ceq]
end

function normal_fit_d(x)
    mu   =  0
    sig  = x
    f1  = normpdf(ted(:,mode),mu,sig)
    f2  = ked(:,mode)
    f = sum((f1-f2).^2);

    return f
end

function normal_fit_r(x)
    mu   =  0
    sig  = x
    f1  = normpdf(ted(:,mode),mu,sig)
    f2  = ked(:,mode)
    f = sum((f1-f2).^2);

    return f
end

function v_diff_d(x)
    vd = Spline1D(x_grid, vvd; k=3)(x) # Cubic spline
    vr = Spline1D(x_grid, vvr; k=3)(x)
    v = -abs(xt_1-x) + lambda*at_1 + beta*(vd*p_d + vr*(1-p_d))-ecost_d

    f = v - vr

    return f
end

function v_diff_r(x)
    vd = Spline1D(x_grid, vvd; k=3)(x)
    vr = Spline1D(x_grid, vvr; k=3)(x)
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

function snp_fit(x)
    x0 = x(1)
    x1 = x(2)
    x2 = x(3)
    x3 = x(4)
    x4 = x(5)
    mu = x(6)
    sig = x(7)
    
    f1  = ((x0 + x1*(t3-mu) + x2*(t3-mu).^2 + x3*(t3-mu).^3 +x4*(t3-mu).^4).^2).*exp(-(t3-mu).^2/sig^2)
    
    if mode == 1
        f2 = kxr
    elseif mode == 2
        f2 = kxd
    elseif mode == 3
        f2 = kar
    elseif mode == 4
        f2 = kad
    end
    
    f = sum((f1-f2).^2)

    return f
end

#= function v_snp_up(x)
    vd = Spline1D(x_grid, vvd; k=1)(x) # Linear spline (interpolation)
    vr = Spline1D(x_grid, vvr; k=1)(x)
    
    fun1 = y -> ((fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2).*exp(-(y-fxr(6)).^2/fxr(7)^2)
    fun2 = y -> (y.*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2).*exp(-(y-fxr(6)).^2/fxr(7)^2)
    
    
    p5r = integral(fun1,u_r(a_i),u_r(a_i)+y_r(a_i))
    g5r = integral(fun2,u_r(a_i),u_r(a_i)+y_r(a_i))
    e5r = g5r/max(p5r,0.00001)
    
    p6r = integral(fun1,u_r(a_i),x)
    g6r = integral(fun2,u_r(a_i),x)
    e6r = g6r/max(p6r,0.00001)
    
    p7r = integral(fun1,x,u_r(a_i)+y_r(a_i))
    g7r = integral(fun2,x,u_r(a_i)+y_r(a_i))
    e7r = g7r/max(p7r,0.00001)
    
    
    if x<u_r(a_i) || x>u_r(a_i) + y_r(a_i)
        f = -abs(e5r-x) + lambda*a_grid(a_i) - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    else 
        f = -abs(e6r-x)*p6r/(p6r+p7r) -abs(e7r-x)*p7r/(p6r+p7r)+ lambda*a_grid(a_i) - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    end

    return f
end

function v_snp_low(x)
    vd = Spline1D(x_grid, vvd; k=1)(x) # Linear spline (interpolation)
    vr = Spline1D(x_grid, vvr; k=1)(x)
    
    fun1 = y -> ((fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2).*exp(-(y-fxr(6)).^2/fxr(7)^2)
    fun2 = y -> (y.*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2).*exp(-(y-fxr(6)).^2/fxr(7)^2)
    
    
    p5r = integral(fun1,l_r(a_i)-y_r(a_i),l_r(a_i))
    g5r = integral(fun2,l_r(a_i)-y_r(a_i),l_r(a_i))
    e5r = g5r/max(p5r,0.00001)
    
    p6r = integral(fun1,l_r(a_i)-y_r(a_i),x)
    g6r = integral(fun2,l_r(a_i)-y_r(a_i),x)
    e6r = g6r/max(p6r,0.00001)
    
    p7r = integral(fun1,x,l_r(a_i))
    g7r = integral(fun2,x,l_r(a_i))
    e7r = g7r/max(p7r,0.00001)
    
    
    if x<l_r(a_i)-y_r(a_i) || x>l_r(a_i)
        f = -abs(e5r-x) + lambda*a_grid(a_i) - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    else 
        f = -abs(e6r-x)*p6r/(p6r+p7r) -abs(e7r-x)*p7r/(p6r+p7r)+ lambda*a_grid(a_i) -ecost_r + beta*(vd*p_d + vr*(1-p_d))
    end

    return f
end

function v_func_snp_ntl(x)
    v_d = x(1)
    v_r = x(2)
    
    f(1) = 0
    
    for k=1:n_app
        if (u_d2(k) >= l_d2(k))
        f1d = y -> (-abs(y-theta)+lambda*a_grid(k)+ beta*v_r).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
        f2d = y -> (-abs(l_d2(k)-theta)+lambda*a_grid(k)-beta*ecost_d)/(1-beta).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
        f3d = y -> (-abs(y-theta)+lambda*a_grid(k)-beta*ecost_d)/(1-beta).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
        f4d = y -> (-abs(u_d2(k)-theta)+lambda*a_grid(k)-beta*ecost_d)/(1-beta).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
        f(1) = f(1) + (integral(f1d,-inf,l_d2(k)-y_d(k)) + integral(f2d,l_d2(k)-y_d(k),l_d2(k)) + integral(f3d,l_d2(k),u_d2(k)) + integral(f4d,u_d2(k),u_d2(k)+y_d(k)) + integral(f1d,u_d2(k)+y_d(k),inf))*pdf_d(k);      
        else     
        f1d = @(y) (-abs(y-theta)+lambda*a_grid(k)+ beta*v_r).*(fxd(1) + fxd(2)*(y-fxd(6)) + fxd(3)*(y-fxd(6)).^2 + fxd(4)*(y-fxd(6)).^3 +fxd(5)*(y-fxd(6)).^4).^2.*exp(-(y-fxd(6)).^2/fxd(7)^2)
        f(1) = f(1) + integral(f1d,-inf,inf)*pdf_d(k)
        end               
    end
    f(1) = f(1) -v_d;
     
    f(2) = 0;
    
    for k=1:n_app
        
        if (u_r2(k) >= l_r2(k))
        
        f1r = @(y) (-abs(y-theta)+lambda*a_grid(k)+beta*v_d).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2) ;
        f2r = @(y) (-abs(l_r2(k)-theta)+lambda*a_grid(k)-beta*ecost_r)/(1-beta).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2) ;
        f3r = @(y) (-abs(y-theta)+lambda*a_grid(k)-beta*ecost_r)/(1-beta).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2) ;
        f4r = @(y) (-abs(u_r2(k)-theta)+lambda*a_grid(k)-beta*ecost_r)/(1-beta).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2) ;
        f(2) = f(2) + (integral(f1r,-inf,l_r2(k)-y_r(k)) + integral(f2r,l_r2(k)-y_r(k),l_r2(k)) ...
                    + integral(f3r,l_r2(k),u_r2(k)) + integral(f4r,u_r2(k),u_r2(k)+y_r(k)) ...
                    + integral(f1r,u_r2(k)+y_r(k),inf))*pdf_r(k);
                
        else
        f1r = @(y) (-abs(y-theta)+lambda*a_grid(k)+beta*v_d).*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2) ;
        f(2) = f(2) + integral(f1r,-inf,inf)*pdf_r(k);        
        
        end        
    end
    
    f(2) = f(2) -v_r

    return f
end     =#



end