# Replication code for Sieg and Yoon (2017)

module Replication_of_Sieg_Yoon_2017

using DataFrames
using CSV

#include("HelperFunctions.jl")

# Options TO BE MADE INTERACTIVE
op_model = 1  # 1 : Baseline model specification / 2 : lambda = 0 / 3 : extended
op_estimation = 0 # 0: simulation of 2nd stage, 1-2 : esimation of 2nd stage (takes long time)
# set the value of parameters
num_sim = 10000 # number of simulation
beta = 0.8      # fixed discount factor
n_app = 5       # number of ability grids
a_max = 1.2     # max ability grids
a_grid = linspace(-a_max,a_max,n_app) # ability grids
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
    std_last(i) = std(ps)
    datafile[:,1+i] = ps/std_last(i)
end

# initial guess
mu_d(2:5) = 0
mu2_d(4:5)= 0
mu_r(2:5) = 0
mu2_r(4:5)= 0

itr = 1
dist = 1

while dist > 1d-10 
    loading_old = [mu_d(2:5) mu2_d(4:5)]
    # factor loading
    # state fixed effects
    y1 = datafile[:,2]
    y2 = datafile[:,3]
    y3 = datafile[:,4]
    y4 = datafile[:,5]
    y5 = datafile[:,6]
    [n_all, ~] = size(y1)
    % estimation of idology using y1 and y2
    Y =  [y1(election_number>1) y2(election_number>1)] # drop 1st term of 2 term governor
    [n,~] = size(Y)
    x = zeros(n,24)
    for i in 1:24
        x(:,i) = (state(election_number>1)==i) # state dummy
    end
    X = cell(n,1)
    for i in 1:n
            X{i} = [x(i,:); x(i,:)*mu_d(2)]
    end
    [beta1,~] = mvregress(X,Y,'algorithm','ecm')
    ideology = zeros(n_all,1)
    for i in 1:24
        ideology(state ==i) = beta1(i)
    end
    residual = zeros(n_all,5)
    residual(:,1) = y1 -ideology
    residual(:,2) = y2 -mu_d(2)*ideology
    y3 = y3 -mu_d(3)*ideology
    y4 = y4 -mu_d(4)*ideology
    y5 = y5 -mu_d(5)*ideology
    Y = [y3(election_number>1) y4(election_number>1) y5(election_number>1)]
    X = cell(n,1)
    for i in 1:n
            X{i} = [x(i,:)  ; x(i,:)*mu2_d(4); x(i,:)*mu2_d(5)]
    end
    [beta2,~] = mvregress(X,Y,'algorithm','ecm')
    ability = zeros(n_all,1)
    for i in 1:24
        ability(state==i) = beta2(i)
    end
    residual(:,3) = y3 -ability
    residual(:,4) = y4 -mu2_d(4)*ability
    residual(:,5) = y5 -mu2_d(5)*ability
    covariance = cov(residual)
    mu_d(2) = covariance(2,3)/covariance(1,3)
    sigma_rho = covariance(1,2)/mu_d(2)
    mu_d(3) = covariance(1,3)/sigma_rho
    mu_d(4) = covariance(1,4)/sigma_rho
    mu_d(5) = covariance(1,5)/sigma_rho
    mu2_d(4) = (covariance(4,5)-mu_d(4)*mu_d(5)*sigma_rho)/(covariance(3,5)-mu_d(3)*mu_d(5)*sigma_rho)
    sigma_a = (covariance(3,4) -mu_d(3)*mu_d(4)*sigma_rho)/mu2_d(4)
    mu2_d(5) = (covariance(3,5)-mu_d(3)*mu_d(5)*sigma_rho)/sigma_a
    mu_r = mu_d;
    mu2_r = mu2_d;
    loading_new = [mu_d(2:5) mu2_d(4:5)];
    dist = max(abs(loading_old-loading_new));
    itr = itr+1;
end

# State names
statename = ["AL";"AZ";"CO";"FL";"GA";"IN";"KS";"KY";"LA";"ME";"MD";"NE";"NJ";"NM";"NC";"OH";"OK";"OR";"PA";"RI";"SC";"SD";"TN";"WV"]

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

d_share = sum(election_number==3 & party2==1)/sum(election_number>=2 & party2==1)
r_share = sum(election_number==3 & party2==2)/sum(election_number>=2 & party2==2)

share_all = zeros(3,4)

for i=1
        ps = residual(:,2+i)
        std4 = std(ps)
        share_all(i,1) = (sum(election_number==3 & ps<-1*std4 ))/(sum(election_number~=2 & ps<-1*std4 ))
        share_all(i,2) = (sum(election_number==3 & ps>=-1*std4 & ps< 0 ))/(sum(election_number~=2 & ps>=-1*std4 & ps< 0 ))
        share_all(i,3) = (sum(election_number==3 & ps>= 0 & ps< 1*std4 ))/(sum(election_number~=2 & ps>= 0 & ps< 1*std4 ))
        share_all(i,4) = (sum(election_number==3 & ps>= 1*std4 ))/(sum(election_number~=2 & ps>= 1*std4 ))
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
        ps = residual(:,i)        
        mean_1(i,j) = mean(ps(election_number==1 & party2==j))
        mean_2(i,j) = mean(ps(election_number==2 & party2==j))
        mean_3(i,j) = mean(ps(election_number==3 & party2==j))        
        mean_all(i,j) = mean(ps(party2==j))
        std_1(i,j) = std(ps(election_number==1 & party2==j))
        std_2(i,j) = std(ps(election_number==2 & party2==j))
        std_3(i,j) = std(ps(election_number==3 & party2==j))        
        std_all(i,j) = std(ps(party2==j))
    end
end

residual2 = residual
for i in 1:5
    residual2(:,i) = residual(:,i)*std_last(i)
end

xlswrite('residual2.xlsx',residual2)

# First Stage Estimation (Kotlarski)
demo  = residual(election_number>=2 & party2==1,:)
repub = residual(election_number>=2 & party2==2,:)
datafile3 = [party2(election_number==1) vote_share(election_number==1) residual(election_number==1,1:2)]
m=10*(floor(10-(-10))+1)       
t3=(linspace(-10,10,m))'
t4=(linspace(-30,30,m))'
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

        b1 = demo(:,1)
        b2 = demo(:,2)/mu_d(2)
        b3 = demo(:,3)-mu_d(3)*b1
        b4 = demo(:,4)/mu2_d(4) - mu_d(4)*b2/mu2_d(4)
        b5 = demo(:,5)/mu2_d(5) - mu_d(5)*b1/mu2_d(5)      
    else      
        Tx=3.0
        Ta=3.0
        T1=3.6
        T2=3.5
        T3=7.5
        T4=7.0
        T5=1.0
 
        b1 = repub(:,1)
        b2 = repub(:,2)/mu_r(2)
        b3 = repub(:,3)-mu_r(3)*b1
        b4 = repub(:,4)/mu2_r(4) - mu_r(4)*b2/mu2_r(4)
        b5 = repub(:,5)/mu2_r(5) - mu_r(5)*b1/mu2_r(5)
    end
    # Ideology
    mx=20*Tx
    ggx=(linspace(-Tx,Tx,mx))'
    [fgx,~,~]=chfc2(b2,b1,Tx,mx)
    # Ability
    ma=20*Ta;
    gga=(linspace(-Ta,Ta,ma))'
    [fga,~,~]=chfc2(b3,b4,Ta,ma)
    # Error terms
    m1=20*T1
    gg1=(linspace(-T1,T1,m1))'
    [~,~,fg1]=chfc2(b2,b1,T1,m1)
    m2=20*T2
    gg2=(linspace(-T2,T2,m2))'
    [~,fg2,~]=chfc2(b2,b1,T2,m2)
    m3=20*T3
    gg3=(linspace(-T3,T3,m3))'
    [~,fg3,~]=chfc2(b3,b4,T3,m3)
    m4=20*T4
    gg4=(linspace(-T4,T4,m4))'
    [~,~,fg4]=chfc2(b3,b4,T4,m4)
    m5=20*T5
    gg5=(linspace(-T5,T5,m5))'
    [~,~,fg5]=chfc3(b3,b4,b5,T5,m5)

    k = 1

    while k <= m
        yx=(1-abs(ggx)/Tx).*real(exp(-1i*ggx*t3(k,1)).*fgx);
        kxm(k,1)=(1/(2*pi))*trapz(ggx,yx);
        
        ya=(1-abs(gga)/Ta).*real(exp(-1i*gga*t3(k,1)).*fga);
        kam(k,1)=(1/(2*pi))*trapz(gga,ya);
        
        y1=(1-abs(gg1)/T1).*real(exp(-1i*gg1*t3(k,1)).*fg1);
        kem(k,1)=(1/(2*pi))*trapz(gg1,y1);
        
        y2=(1-abs(gg2)/T2).*real(exp(-1i*gg2*t3(k,1)).*fg2);
        kem(k,2)=(1/(2*pi))*trapz(gg2,y2);
        
        y3=(1-abs(gg3)/T3).*real(exp(-1i*gg3*t3(k,1)).*fg3);
        kem(k,3)=(1/(2*pi))*trapz(gg3,y3);
        
        y4=(1-abs(gg4)/T4).*real(exp(-1i*gg4*t3(k,1)).*fg4);
        kem(k,4)=(1/(2*pi))*trapz(gg4,y4);
        
        y5=(1-abs(gg5)/T5).*real(exp(-1i*gg5*t4(k,1)).*fg5);
        kem(k,5)=(1/(2*pi))*trapz(gg5,y5);
        
            
    k=k+1;
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

end