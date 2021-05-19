module code_find_standard_ttl
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017

# Solving for election standards and valufuction of median voter in 2-term limit case
# Estimated densities
x0[1] = -4.7
x0[2] = -4.7
x0[3:3+n_app-1] = 0.2
x0[3+n_app:3+2*n_app-1]   = -0.2
x0[3+2*n_app:3+3*n_app-1] = 0.2
x0[3+3*n_app:3+4*n_app-1] = -0.2

options = optimoptions('fsolve','TolFun',1.0d-10,'Display','off')
[x,fval] = fsolve(@myfun_ttl,x0,options)
n = n_app

u_d = x[3:3+1*n-1]
l_d = x[3+1*n:3+2*n-1]
u_r = x[3+2*n:3+3*n-1]
l_r = x[3+3*n:3+4*n-1]

end