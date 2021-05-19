module fun_chfc2
export chfc2

function chfc2(v1, v2, T, m)
    tol = 0.01
    trace = 0
    t1 = collect(LinRange(-T,T,m))
    f1 = ones(m, 1)
    f2 = ones(m, 1)
    f3 = ones(m, 1)
    k = 1
    while k <= m
        p = 0
        c1 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        p = 1
        c2 = quad(@cch2,0,t1(k,1),tol,trace,v1,v2,p)
        f1(k, 1) = exp(c1+1i*c2)
        (w, v, dw) = chf1(t1(k,1), v1, v2)
        f2(k, 1) = v/f1(k, 1)    
        f3(k, 1) = w/f1(k, 1)
        k = k + 1
    end

    return [f1, f2, f3]
end

end