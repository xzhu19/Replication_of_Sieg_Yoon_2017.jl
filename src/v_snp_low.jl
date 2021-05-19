module fun_v_snp_low

function v_snp_low(x)
    global beta p_d vvd vvr x_grid a_i lambda a_grid ecost_r
    global fxr
    global l_r y_r
    
    vd = interp1(x_grid,vvd,x)
    vr = interp1(x_grid,vvr,x)
    
    fun1 = y -> (fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2)
    fun2 = y -> y.*(fxr(1) + fxr(2)*(y-fxr(6)) + fxr(3)*(y-fxr(6)).^2 + fxr(4)*(y-fxr(6)).^3 +fxr(5)*(y-fxr(6)).^4).^2.*exp(-(y-fxr(6)).^2/fxr(7)^2)
    
    p5r = integral(fun1,l_r(a_i)-y_r(a_i),l_r(a_i))
    g5r = integral(fun2,l_r(a_i)-y_r(a_i),l_r(a_i))
    e5r = g5r/max(p5r, 0.00001)
    
    p6r = integral(fun1,l_r(a_i)-y_r(a_i),x)
    g6r = integral(fun2,l_r(a_i)-y_r(a_i),x)
    e6r = g6r/max(p6r, 0.00001)
    
    p7r = integral(fun1,x,l_r(a_i))
    g7r = integral(fun2,x,l_r(a_i))
    e7r = g7r/max(p7r, 0.00001)
    
    if x<l_r(a_i)-y_r(a_i) || x>l_r(a_i)
        f = -abs(e5r-x) + lambda*a_grid(a_i) - ecost_r + beta*(vd*p_d + vr*(1-p_d))
    else 
        f = -abs(e6r-x)*p6r/(p6r+p7r) -abs(e7r-x)*p7r/(p6r+p7r)+ lambda*a_grid(a_i) -ecost_r + beta*(vd*p_d + vr*(1-p_d))
    end

    return f
end

end