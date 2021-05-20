module fun_normal_fit_r
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export normal_fit_r

using Distributions

function normal_fit_r(x)
    mu   =  0
    sig  = x
    f1 = pdf(Normal(mu, sig), ter[:, mode])
    f2  = ker[:, mode]
    f = sum((f1-f2).^2)

    return f
end

end