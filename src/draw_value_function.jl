module code_draw_value_function

global xt_1 at_1 a_i
global theta p_d x_grid vvd vvr
global m_grid 

m_grid = 100
x_grid = transpose(collect(LinRange(-7,7,m_grid)))
vvd = transpose(collect(LinRange(-10,10,m_grid)))
vvr = transpose(collect(LinRange(-10,10,m_grid)))

options = optimoptions('fsolve','Display','none');

for i in 1:m_grid
    theta = x_grid(i)
    x0 = [-10;-10]
    [x,fval]= fsolve(@v_func_snp,x0,options)
    vvd[i] = x[1]
    vvr[i] = x[2]
end

rho_grid = transpose(collect(LinRange(-0.4,-0.2,9)))

grid = transpose(collect(LinRange(-1.5,1.5,101)))
vv = zeros(2,101)
v = zeros(2,101)
w = zeros(2,101)

x0 = [-10; -10]

for i in 1:101
    theta = grid[i]
    [x,fval]= fsolve(@v_func_snp,x0,options)
    vv[1,i] = x[1]
    vv[2,i] = x[2]
    x0[1] = x[1]
    x0[2] = x[2]
end

at_1 = 0
xt_1 = -0.4
for i in 1:101
    v(1,i) = v_snp_r(grid(i))
end

at_1 = 0
xt_1 = -0.05
for i in 1:101
    v(2,i) = v_snp_r(grid(i))
end

figure
plot(grid,vv(1,:),'Color','k','LineStyle','--','LineWidth',2)
hold on
plot(grid,v(1,:),'Color','r','LineStyle','-.','LineWidth',2)
hold on
plot(grid,v(2,:),'Color','b','LineStyle','-','LineWidth',2)
hleg = legend('$V^D(\theta)$','$V^{I,R}(\theta,-0.4,0)$','$V^{I,R}(\theta,-0.05,0)$');
set(hleg, 'Box','off','Location','NorthEast','Interpreter','latex')
ylabel('value function')
xlabel('ideological location (\theta)')
axis([-1.5 1.5 -9 -2])

a_i = 3
for i in 1:101
    w(1,i) = v_snp_up(grid[i])
end

a_i = 3
for i in 1:101
    w(2,i) = v_snp_low(grid[i])
end


figure
plot(grid,vv(1,:),'Color','b','LineStyle','-','LineWidth',2)
hold on
plot(grid,w(1,:),'Color','r','LineStyle','--','LineWidth',2)
hleg = legend('$V^D(\theta)$','$V^{I,R}(\theta,\bar{s}_R(a),a)$');
set(hleg, 'Box','off','Location','NorthEast','Interpreter','latex')
ylabel('value function')
xlabel('ideological location (\theta)')
axis([-1.5 1.5 -9 -3])

figure
plot(grid,vv(1,:),'Color','b','LineStyle','-','LineWidth',2)
hold on
plot(grid,w(2,:),'Color','r','LineStyle','--','LineWidth',2)
hleg = legend('$V^D(\theta)$','$V^{I,R}(\theta,\underline{s}_R(a),a)$');
set(hleg, 'Box','off','Location','NorthEast','Interpreter','latex')
ylabel('value function')
xlabel('ideological location (\theta)')
axis([-1.5 1.5 -9 -3])

vv0=p_d*vv(1,:) + (1-p_d)*vv(2,:)

end