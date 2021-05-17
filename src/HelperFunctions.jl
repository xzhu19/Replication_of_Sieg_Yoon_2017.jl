# Replication code for Sieg and Yoon (2017)

module HelperFunctions 

using Plots
using NLopt
using Random
using LinearAlgebra
using Dierckx

function chf1(t2, v1, v2)
    n = size(v1, 1)
    k1 = size(v1, 2)
    k2 = size(v2, 2)
    s2 = size(t2)

    if s2(1, 1) > s2(1, 2)
        t2 = transpose(t2)
        d = s2(1, 1)
    else 
        d = s2(1, 2)
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
            w0 = sum(exp(1i*kron(g2,t2).*kron(g1,v2(:,l2))),1)
            v0 = sum(exp(1i*kron(g2,t2).*kron(g1,v1(:,l1))),1)
            dw0 = sum(1i*kron(g1,v1(:,l1)).*exp(1i*kron(g2,t2).*kron(g1,v2(:,l2))),1)
            dw = dw + dw0
            w = w + w0
            v = v + v0
            l2 = l2 + 1
        end
        l1 = l1 + 1
    end

    w = w/(n*k1*k2)
    v = v/(n*k1*k2)
    dw = dw/(n*k1*k2)

    if s2(1, 1) > s2(1, 2)
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

end