module code_find_standard_ntl

# Solving for election standards and valufuction of median voter in no term limit case
global n_app 
# global u_d2 l_d2 u_r2 l_r2 
# global a_grid

# Define matrix

x0[1] = -3.5
x0[2] = -3.5

# Solving for system of equations
options = optimoptions('fsolve','TolFun',1.0e-10,'Display','off')
[x,fval] = fsolve(@myfun_ntl2,x0,options)

# Update election standards
n = n_app

end