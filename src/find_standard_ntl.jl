module code_find_standard_ntl
include("./Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017

# Solving for election standards and valufuction of median voter in no term limit case
# Define matrix
x0[1] = -3.5
x0[2] = -3.5

# Solving for system of equations
options = optimoptions('fsolve','TolFun',1.0e-10,'Display','off')
[x,fval] = fsolve(@myfun_ntl2,x0,options)

# Update election standards
n = n_app

end