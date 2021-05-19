module fun_snp_fit
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export snp_fit

function snp_fit(x)
    x0 = x[1]
    x1 = x[2]
    x2 = x[3]
    x3 = x[4]
    x4 = x[5]
    mu = x[6]
    sig = x[7]
    
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

end