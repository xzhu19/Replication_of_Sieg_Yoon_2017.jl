# Replication code for Sieg and Yoon (2017)

module Replication_of_Sieg_Yoon_2017

export ter, ker, mode, ted, ked, t3, kxr, kxd, kar,kad

using DataFrames
using CSV
using Statistics
using Plots
using GLM
using NLopt, Optim
using Trapz
using LinearAlgebra
using JuMP
using Distributions

include("./chf1.jl")
include("./cch2.jl")
include("./chfc2.jl")
include("./chfc3.jl")
include("./mycon.jl")

using .fun_chf1, .fun_cch2, .fun_chfc2, .fun_chfc3, .fun_mycon

#include("HelperFunctions.jl")

# Options TO BE MADE INTERACTIVE
op_model = 1  # 1 : Baseline model specification / 2 : lambda = 0 / 3 : extended
op_estimation = 0 # 0: simulation of 2nd stage, 1-2 : esimation of 2nd stage (takes long time)

# Set the value of parameters
num_sim = 10000 # Number of simulations
beta = 0.8      # Fixed discount factor
n_app = 5       # Number of ability grids
a_max = 1.2     # Max ability grids
a_grid = transpose(collect(LinRange(-a_max,a_max,n_app))) # Ability grids
num_eval = 5000 

# Read data
datafile = CSV.read("/Users/Victor/Documents/Final project Numerical Methods/Replication_of_Sieg_Yoon_2017/src/datafile6.csv", DataFrame; header=false)
party2 = datafile[:,1]
election_number = datafile[:,7]
election_code = datafile[:,8]
vote_share = datafile[:,9]
state = datafile[:,10]
# Normalize data
std_last = zeros(5, 1)
for i in 1:5     
    ps = datafile[:,1+i]        
    std_last[i] = std(ps)
    datafile[:,1+i] = ps/std_last[i]
end

# initial guess
mu_d = [0, 0, 0, 0, 0]
mu2_d = [0, 0, 0, 0, 0]
mu_r = [0, 0, 0, 0, 0]
mu2_r = [0, 0, 0, 0, 0]

itr = 1
dist = 1

while dist > 1e-10 
    loading_old = [mu_d[2], mu_d[3], mu_d[4], mu_d[5], mu2_d[4], mu2_d[5]]
    # factor loading
    # state fixed effects
    y1 = datafile[:,2]
    y2 = datafile[:,3]
    y3 = datafile[:,4]
    y4 = datafile[:,5]
    y5 = datafile[:,6]
    n_all = size(y1)[1]
    # Estimation of idology using y1 and y2
    Y = hcat(datafile[election_number.>1, 2], datafile[election_number.>1, 3]) # Drop 1st term of 2-term governors
    n = size(Y)[1]
    x = zeros(n, 24)
    for i in 1:24
            x[:,i] = datafile[election_number.>1, 10] .== i # state dummy
    end
    X = Vector{Any}(undef, n)
    for i in 1:n
            X[i] = [x[i,:], x[i,:]*mu_d[2]]
    end
    beta1 = mvregress(X,Y,'algorithm','ecm')
    ideology = zeros(n_all, 1)
    for i in 1:24
        ideology(state == i) = beta1[i]
    end
    residual = zeros(n_all, 5)
    residual[:,1] = y1 - ideology
    residual[:,2] = y2 - mu_d[2]*ideology
    y3 = y3 - mu_d[3]*ideology
    y4 = y4 - mu_d[4]*ideology
    y5 = y5 - mu_d[5]*ideology
    Y = [y3(election_number>1) y4(election_number>1) y5(election_number>1)]
    X = Vector{Any}(undef, n)
    for i in 1:n
            X[i] = [x(i,:), x(i,:)*mu2_d(4), x(i,:)*mu2_d(5)]
    end
    [beta2,~] = mvregress(X,Y,'algorithm','ecm')
    ability = zeros(n_all, 1)
    for i in 1:24
        ability[state == i] = beta2[i]
    end
    residual[:,3] = y3 - ability
    residual[:,4] = y4 - mu2_d[4]*ability
    residual[:,5] = y5 - mu2_d[5]*ability
    covariance = cov(residual)
    mu_d[2] = covariance[2,3]/covariance[1,3]
    sigma_rho = covariance[1,2]/mu_d[2]
    mu_d[3] = covariance[1,3]/sigma_rho
    mu_d[4] = covariance[1,4]/sigma_rho
    mu_d[5] = covariance[1,5]/sigma_rho
    mu2_d[4] = (covariance[4,5]-mu_d[4]*mu_d[5]*sigma_rho)/(covariance[3,5]-mu_d[3]*mu_d[5]*sigma_rho)
    sigma_a = (covariance[3,4] -mu_d[3]*mu_d[4]*sigma_rho)/mu2_d[4]
    mu2_d[5] = (covariance[3,5]-mu_d[3]*mu_d[5]*sigma_rho)/sigma_a
    mu_r = mu_d
    mu2_r = mu2_d
    loading_new = [mu_d[2], mu_d[3], mu_d[4], mu_d[5], mu2_d[4], mu2_d[5]]
    dist = max(abs(loading_old - loading_new))
    itr = itr + 1
end

# State names: those limited to 2 consecutive terms (NM and OR changed it)
statename = ["AL"; "AZ"; "CO"; "FL"; "GA"; "IN"; "KS"; "KY"; "LA"; "ME"; "MD"; "NE"; "NJ"; "NM"; "NC"; "OH"; "OK"; "OR"; "PA"; "RI"; "SC"; "SD"; "TN"; "WV"]

hFig = figure(1)
set(hFig, 'Position', [100 100 1000 400])
celldata = cellstr(statename)
scatter(beta1,beta2)
dx = 0.02
dy = 0.02 # displacement so the text does not overlay the data points
text(beta1+dx, beta2+dy, celldata)
axis([-2 2 -0.8 0.8])
ylabel('competence')
xlabel('ideology')

# Data moments

d_share = sum(datafile[election_number.==3 & party2.==1, 1], datafile[election_number.==3 & party2.==1, 7])/sum(atafile[election_number.>1 & party2.==1, 1], datafile[election_number.>1 & party2.==1, 7])
d_share = sum(election_number==3 & party2==1)/sum(election_number>=2 & party2==1)
r_share = sum(election_number==3 & party2==2)/sum(election_number>=2 & party2==2)

share_all = zeros(3,4)

for i = 1
        ps = residual[:, 3]
        std4 = std(ps)
        share_all[i,1] = (sum(election_number==3 & ps<-1*std4 ))/(sum(election_number!=2 & ps<-1*std4 ))
        share_all[i,2] = (sum(election_number==3 & ps>=-1*std4 & ps< 0 ))/(sum(election_number!=2 & ps>=-1*std4 & ps< 0 ))
        share_all[i,3] = (sum(election_number==3 & ps>= 0 & ps< 1*std4 ))/(sum(election_number!=2 & ps>= 0 & ps< 1*std4 ))
        share_all[i,4] = (sum(election_number==3 & ps>= 1*std4 ))/(sum(election_number!=2 & ps>= 1*std4 ))
end

mean_1 = zeros(5,2)
mean_2 = zeros(5,2)
mean_3 = zeros(5,2)
mean_all = zeros(5,2)
std_1 = zeros(5,2)
std_2 = zeros(5,2)
std_3 = zeros(5,2)
std_all = zeros(5,2)

for i in 1:5
    for j in 1:2       
        ps = residual[:,i]        
        mean_1[i,j] = mean(ps(election_number==1 & party2==j))
        mean_2[i,j] = mean(ps(election_number==2 & party2==j))
        mean_3[i,j] = mean(ps(election_number==3 & party2==j))        
        mean_all[i,j] = mean(ps(party2==j))
        std_1[i,j] = std(ps(election_number==1 & party2==j))
        std_2[i,j] = std(ps(election_number==2 & party2==j))
        std_3[i,j] = std(ps(election_number==3 & party2==j))        
        std_all(i,j] = std(ps(party2==j))
    end
end

residual2 = residual

for i in 1:5
    residual2[:,i] = residual[:,i]*std_last[i]
end

# First Stage Estimation (Kotlarski)
demo  = residual[election_number>=2 & party2==1, :]
repub = residual[election_number>=2 & party2==2, :]
datafile3 = [party2(election_number==1) vote_share(election_number==1) residual(election_number==1,1:2)]
m = 10*(floor(10-(-10))+1)       
t3 = collect(LinRange(-10,10,m))
t4 = collect(LinRange(-30,30,m))
kxm = zeros(m,1)
kam = zeros(m,1)
kem = zeros(m,5)
ted =[t3 t3 t3 t3 t4]
ter =[t3 t3 t3 t3 t4]

for kk in 1:2
    if kk == 1  
        Tx=3.0
        Ta=3.0
        T1=3.6
        T2=3.5
        T3=7.5
        T4=7.0
        T5=1.0

        b1 = demo[:,1]
        b2 = demo[:,2]/mu_d[2]
        b3 = demo[:,3]-mu_d[3]*b1
        b4 = demo[:,4]/mu2_d[4] - mu_d[4]*b2/mu2_d[4]
        b5 = demo[:,5]/mu2_d[5] - mu_d[5]*b1/mu2_d[5]      
    else      
        Tx=3.0
        Ta=3.0
        T1=3.6
        T2=3.5
        T3=7.5
        T4=7.0
        T5=1.0
 
        b1 = repub[:,1]
        b2 = repub[:,2]/mu_r[2]
        b3 = repub[:,3]-mu_r[3]*b1
        b4 = repub[:,4]/mu2_r[4] - mu_r[4]*b2/mu2_r[4]
        b5 = repub[:,5]/mu2_r[5] - mu_r[5]*b1/mu2_r[5]
    end
    # Ideology
    mx = 20*Tx
    ggx = collect(LinRange(-Tx,Tx,mx)))
    fgx = chfc2(b2,b1,Tx,mx)[1]
    # Ability
    ma = 20*Ta
    gga = collect(LinRange(-Ta,Ta,ma))
    fga = chfc2(b3,b4,Ta,ma)[1]
    # Error terms
    m1 = 20*T1
    gg1 = collect(LinRange(-T1,T1,m1))
    fg1 = chfc2(b2,b1,T1,m1)[3]
    m2 = 20*T2
    gg2 = collect(LinRange(-T2,T2,m2))
    fg2 = chfc2(b2,b1,T2,m2)[2]
    m3 = 20*T3
    gg3 = collect(LinRange(-T3,T3,m3))
    fg3 = chfc2(b3,b4,T3,m3)[2]
    m4 = 20*T4
    gg4 = collect(LinRange(-T4,T4,m4))
    fg4 = chfc2(b3,b4,T4,m4)[3]
    m5 = 20*T5
    gg5 = collect(LinRange(-T5,T5,m5))
    fg5 = chfc3(b3,b4,b5,T5,m5)[3]

    k = 1

    while k <= m
        yx = (1-abs(ggx)/Tx).*real(exp(-1i*ggx*t3[k,1]).*fgx)
        kxm[k,1] = (1/(2*pi))*trapz(ggx,yx)
        
        ya = (1-abs(gga)/Ta).*real(exp(-1i*gga*t3[k,1]).*fga)
        kam[k,1] = (1/(2*pi))*trapz(gga,ya)
        
        y1 = (1-abs(gg1)/T1).*real(exp(-1i*gg1*t3[k,1]).*fg1)
        kem[k,1] = (1/(2*pi))*trapz(gg1,y1)
        
        y2 = (1-abs(gg2)/T2).*real(exp(-1i*gg2*t3[k,1]).*fg2)
        kem[k,2] = (1/(2*pi))*trapz(gg2,y2)
        
        y3 = (1-abs(gg3)/T3).*real(exp(-1i*gg3*t3[k,1]).*fg3)
        kem[k,3] = (1/(2*pi))*trapz(gg3, y3)
        
        y4 = (1-abs(gg4)/T4).*real(exp(-1i*gg4*t3[k,1]).*fg4)
        kem[k,4] = (1/(2*pi))*trapz(gg4, y4)
        
        y5 = (1-abs(gg5)/T5).*real(exp(-1i*gg5*t4[k,1]).*fg5)
        kem[k,5] = (1/(2*pi))*trapz(gg5, y5)
        
        k = k + 1
    end

    if kk == 1
        kxd = kxm
        kad = kam
        ked = kem
    else
        kxr = kxm
        kar = kam
        ker = kem
    end
end

# Approximate distibution of ideology and competence using SNP density
options = optimoptions('fmincon','display','off')
A = []b = []Aeq = []beq = []lb = []ub = []
xx_vector = zeros(4,7)
x0=[0.5880    0.0087   -0.0617   -0.0004    0.0048   -0.2016    2.0044
    0.6078   -0.0429   -0.0723    0.0026    0.0077    0.2638    1.7891
   -0.7009    0.0033    0.1973   -0.0001   -0.0083   -0.0504    2.1520
    0.7060    0.0070   -0.2045   -0.0003    0.0089   -0.0088    2.1206]
for mode in 1:4  
    xx_vector[mode,:] = fmincon(@snp_fit,x0[mode,:],A,b,Aeq,beq,lb,ub,@mycon,options)
end
fxr = xx_vector[1, :]
fxd = xx_vector[2, :]
far = xx_vector[3, :]
fad = xx_vector[4, :]
# Approximate distibution of error terms using Normal Density
options = optimset('display','off')
x0v = [1 1 1 1 15]

sigr = zeros(5,1)
sigd = zeros(5,1)

for i in 1:5
    mode = i
    x0 = x0v[i]
    sigr[i] = fminsearch(@normal_fit_r,x0,options)
    sigd[i] = fminsearch(@normal_fit_d,x0,options)
end

sigd[2] = sigd[2]*mu_d[2]
sigd[3] = sqrt(sigd[3]^2-(mu_d[3]*sigd[1])^2)
sigd[4] = sqrt((mu2_d[4]*sigd[4])^2-(mu_d[4]*sigd[2]/mu_d[2])^2)
sigd[5] = sqrt((mu2_d[5]*sigd[5])^2-(mu_d[5]*sigd[1])^2)
sigr[2] = sigr[2]*mu_r[2]
sigr[3] = sqrt(sigr[3]^2-(mu_r[3]*sigr[1])^2)
sigr[4] = sqrt((mu2_r[4]*sigr[4])^2-(mu_r[4]*sigr[2]/mu_r[2])^2)
sigr[5] = sqrt((mu2_r[5]*sigr[5])^2-(mu_r[5]*sigr[1])^2)

# Approximate distribution of competence using discrete grids
f15  = y -> (far[1] + far[2]*(y-far[6]) + far[3]*(y-far[6]).^2 + far[4]*(y-far[6]).^3 +far[5]*(y-far[6]).^4).^2.*exp(-(y-far[6]).^2/far[7]^2)
f16  = y -> (fad[1] + fad[2]*(y-fad[6]) + fad[3]*(y-fad[6]).^2 + fad[4]*(y-fad[6]).^3 +fad[5]*(y-fad[6]).^4).^2.*exp(-(y-fad[6]).^2/fad[7]^2)

cdf_r = transpose(collect(LinRange(0, 1, n_app+1)))
cdf_d = transpose(collect(LinRange(0, 1, n_app+1)))
a_dist = a_grid[2] - a_grid[1]
for i in 2:n_app
    cdf_r[i] = integral(f15,-inf,a_grid[i-1]+a_dist*0.5)
    cdf_d[i] = integral(f16,-inf,a_grid[i-1]+a_dist*0.5)
end
pdf_r = cdf_r[2:n_app+1] - cdf_r[1:n_app]
pdf_d = cdf_d[2:n_app+1] - cdf_d[1:n_app]

# Plots
fun1 = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
fun3 = y -> (fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)

figure
plot(t3,fun1(t3),'Color','r','LineStyle','-','LineWidth',2)
hold on
plot(t3,fun3(t3),'Color','b','LineStyle','--','LineWidth',2)
hleg = legend('Republican','Democrat')
set(hleg, 'Box','off','Location','NorthEast')
axis([-4 4 0 0.6])
xlabel('ideology')

figure
plot(t3,f15(t3),'Color','r','LineStyle','-','LineWidth',2)
hold on
plot(t3,f16(t3),'Color','b','LineStyle','--','LineWidth',2)
hleg = legend('Republican','Democrat')
set(hleg, 'Box','off','Location','NorthEast')
axis([-4 4 0 0.6])
xlabel ('competence')

# 2nd stage using SMM

w_diag = [0.0280439; 0.0302329; 0.0473946; 0.1031027; 0.0809237; 0.0704786; 0.0344337; 0.0301466; 0.0650172]
weight = inv(diag(w_diag.^2))
    
if op_model == 1
    load('test1.mat')
elseif op_model == 2
    load('test2.mat')
else
    load('test3.mat')
end

x0=xx

disp('first stage estimate')
disp(mu_d[2:5])
disp(mu2_d[4:5])

op_policyexp        = 0
op_thirdstage       = 0 
op_valuefunction    = 0 
op_welfare          = 0 
    
if op_estimation==0 && op_model==1
    op_policyexp        = 1 
    op_thirdstage       = 1 
    op_valuefunction    = 1 
    op_welfare          = 1 
end

if op_estimation == 1
    op_print_results    = 0
    options = optimset('Display','iter','MaxFunEvals',num_eval)
    [xx,fval] = fminsearch(@smm12,x0,options)
    save test1.mat xx
    op_print_results = 1
    smm12(xx)
elseif op_estimation==2
    op_print_results    = 0
    options = psoptimset('Display','iter','MaxFunEvals',num_eval,'TolX',1d-6)
    A =[]    b = []    Aeq = []    beq = []
    LB = x0 - abs(x0)*0.2
    UB = x0 + abs(x0)*0.2
    xx = patternsearch(@smm12,x0,A,b,Aeq,beq,LB,UB,options)
    save test1.mat xx
    op_print_results = 1
    smm12(xx)
else
    disp('2nd stage estimate')
    disp(xx)
    op_print_results = 1
    smm12(x0)
end

end