module fun_normal_fit_r
using Replication_of_Sieg_Yoon_2017

function normal_fit_r(x)
    mu   =  0
    sig  = x
    f1  = normpdf(ted[:,mode],mu,sig)
    f2  = ked[:,mode]
    f = sum((f1-f2).^2)

    return f
end

end