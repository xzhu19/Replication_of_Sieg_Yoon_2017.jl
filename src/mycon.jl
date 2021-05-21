module fun_mycon
export mycon

function mycon(x)
    x0 = x[1]
    x1 = x[2]
    x2 = x[3]
    x3 = x[4]
    x4 = x[5]
    mu = x[6]
    sig = x[7]
    
    c = []
    ceq = x0^2*sqrt(pi)*sig + (x1^2+2*x0*x2)*sqrt(pi)*sig^3/2 + (x2^2 + 2*x0*x4 + 2*x1*x3)*sqrt(pi)*sig^5*3/4 + (x3^2 + 2*x2*x4)*sqrt(pi)*sig^7*15/8 + (x4^2)*sqrt(pi)*sig^9*105/16 - 1

    return [c, ceq]
end

end