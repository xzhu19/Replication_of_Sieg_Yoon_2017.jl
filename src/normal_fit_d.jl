module fun_normal_fit_d
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export normal_fit_d

using Distributions

function normal_fit_d(x)
    mu   =  0
    sig  = x
    f1 = pdf(Normal(mu, sig), ted[:, mode])
    f2  = ked[:, mode]
    f = sum((f1-f2).^2)

    return f
end

end