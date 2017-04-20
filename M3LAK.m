function [accuracy] = M3LAK(numclass,numview,C,m,X,Y,xtest,ytest,numberOfIterations,numitermean,Ntest)
% numclass: the number of classes
% numview: the number of views
% C: the regularization patameter
% m: the subspace dimension
% X: X{i} D*N multi-view training data 
% Y: the training label vector N*1
% Ntest: the number of testing data
% xtest: xtest{i} D*Ntest multi-view training data 
% ytest: the testing label vector Ntest*1
% numberOfIterations: the number of MCMC iterations for training and testing
% numitermean: the number MCMC iterations for prediction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H.Psi0 = eye(m)*1e+0;
H.nv0 = 1e+0*m;
H.k0 = 1e+0;
H.mu0 = 1e-2*ones(m,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 100; % M 250
K0 = m;

Kc=K0;  
for i = 1:numview
    [D(i),N] = size(X{i});%
     K(i) = 1;
     Kc = Kc + K(i);
end

a_tau0 = 1e-2;
b_tau0 = 1e-5;
a_r0 = 1e-1;%10^(-3)
b_r0 = 1e-5;%10^(-3)

for i = 1:numview
    for k = 1:K0
        r(i,k) = 1;%gamrnd(a_r0, 1/b_r0); % r W covariance
        W{i}(:,k) = 1e-3*ones(D(i),1);%mvnrnd(zeros(1,D(i)),eye(D(i))./r(i,k))'; %    
    end
    for k=1:K(i)
        r_v{i}(k) = 1e+3;%fuzhu variabel covariance smaller 
        V{i}(:,k) = 1e-3*ones(D(i),1);
    end
    tau(i) = 1;%gamrnd(a_tau0, 1/b_tau0);
    
end

for n = 1:N
    S(:,n) = 1e-3*ones(K0,1);%mvnrnd(zeros(1,K0),eye(K0))' ;
    for i = 1:numview
        S_addition{i}(:,n) = 1e-3*ones(K(i),1);
    end
end
covriance_S_addition = 1e+3;
for n = 1:Ntest
    S_test(:,n) = 1e-3*ones(K0,1);%mvnrnd(zeros(1,K0),eye(K0))' ;
    for i = 1:numview
        S_addition_test{i}(:,n) = 1e-3*ones(K(i),1);
    end
end

%%%Large-Margin
for c = 1:numclass
a_v0 = 1e-1;
b_v0 = 1e-5;
v{c} = 0.01;%gamrnd(a_v0, 1/b_v0); % v beta covariance????????????gamma???????0??????
Beta{c} = 1e-3*ones(2*M+1,1);%mvnrnd(zeros(1,2*M+1),eye(2*M+1)./v{c})'; % (2*M+1)*1;
Sigma_beta{c} = eye(2*M+1);%eye(2*M+1)./v{c};%
mu_beta{c} = zeros(2*M+1,1);%
inv_Sigma_beta{c} = eye(2*M+1);%v{c}.*eye(2*M+1);%
Lambda{c} = 1e-3*ones(N,1);
end
%%%HMC parameter
%alpha is the concentration parameter of a DP. alphaa and alphab are the shape and rate parameters of the Gamma prior on alpha
alphaa= 0.5;%no use
alphab= 0.5;%no use
alpha = 1;%
% thetaStar holds the unique parameters sampled from the Dirichlet process
varpi(1) = betarnd(1,alpha); %varpi contains component weights
varpi_star = 1-varpi(1);
slice_u = unifrnd(0,varpi(1),M,1); %the introduced slice variables 
slice_u_star = min(slice_u);

newJ = ones(M, 1); % newJ contains component index for each data
newNj = M; % newNj contains the number of data in each component
% balance the reducing dimension and the classifier 0.1 for parkinsons data set
Leapfrog = 2;
%%%%%%%%
thetaStar(1).Sigma = iwishrnd(H.Psi0,H.nv0);
thetaStar(1).mu = mvnrnd(H.mu0',(((thetaStar(1).Sigma)./H.k0)+((thetaStar(1).Sigma)./H.k0)')./2)';
for l = 1:M
    G(:,l) = zeros(m,1);%mvnrnd((thetaStar(1).mu)',(thetaStar(1).Sigma+thetaStar(1).Sigma')./2)';% m*M
end%zeros(m,1);%
thetaStar(1).invSigma = inv(thetaStar(1).Sigma);
thetaStar(1).detSigma = det(thetaStar(1).Sigma);
Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)]; % (2*M+1)*N
%%%%%%%
pro =0;
%%%%

for iter = 1:numberOfIterations

     iter
     epsilons = 1/(iter+100);
     epsilong = 1/(iter+100);

    %sampling s, using HMC methods 
    [S] = sampleS_Modified_CCA_BNK(tau,W,V,X,C,Y,Beta,Lambda,K0,M,S,N,numview,S_addition, numclass, epsilons, Leapfrog, G);
    
    [S_addition] = sampleS_addition_Modified_CCA_BNK(N, numview, K, S, V, tau, X, W, covriance_S_addition);
    
    Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)]; % (2*M+1)*N
    
    [W] = sampleW_Modified_CCA(numview, r, S_addition, tau, X, D, W, V, K0, S);
    
    [V] = sampleV_Modified_CCA(numview, r_v, S_addition, tau, X, D, W, V, K, S);
    
    [r] = sampleR_Modified_CCA(a_r0, b_r0, W, numview, D, K0);
    
    [tau] = sampleTau_Modified_CCA(a_tau0, b_tau0, S, X, W, N, D, numview, V, S_addition);

    %%%Large Margin
      [Lambda] = sampleLambda_CCA(N,C,Phi_S,Beta,Y,numclass);
      
      [Beta,~,~]  = sampleBeta_nogamma_CCA(v,C,Lambda,Phi_S,Y,M,numclass);

      %sampling z, mu_z,Sigma_z, using Distributed inference for DPM
      [newJ,newNj, slice_u_star, varpi, varpi_star, thetaStar,slice_u] = sampleDPM(newJ,newNj, slice_u_star, varpi, varpi_star, alpha, H, M, G,thetaStar,slice_u,alphaa, alphab);
      %disp(newNj')
      

       Z = newJ;

      [G] =  sampleG_CCA_quick(N,M,G,C,Y,Beta,Lambda,Z,thetaStar,S,epsilong,numclass,Leapfrog,Phi_S);
 
       Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)];
    %%Phi = V*V'+tau^(-1)*I;
     if iter>numitermean
     [S_test] = predictS_Phi_Modified_CCA(tau,W,V,xtest,K0,S_test,Ntest,numview);
     Phi_S_test = [1/sqrt(M)*[cos(G'*S_test);sin(G'*S_test)];ones(1,Ntest)];
       for c=1:numclass
        test{c}=Phi_S_test'*(Beta{c});
       end
     ac = 0;
     for j = 1:Ntest
         teste = zeros(numclass,1); 
         for c=1:numclass
             teste(c)=test{c}(j);
         end
        testgroup(j) = sign(test{c}(j));
        if testgroup(j)==ytest(j)
            ac=ac+1;
        end
     end
     ac_observe=ac/Ntest
     pro = sign(Phi_S_test'*(Beta{1})) + pro;  
     end 
end
accuracy=mean(sign(pro)==ytest);