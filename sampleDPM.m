% sampling z, mu_z,Sigma_z, using Distributed inference for DPM
% newJ contains component index for each data
% newNj contains the number of data in each component
% varpi(1) = betarnd(1,alpha) %varpi contains component weights 初始化的shih 
% varpi_star = 1-varpi(1);抽取新component的概率
% slice_u = unifrnd(0,varpi(1),n,1); %the introduced slice variables 
function [newJ,newNj, slice_u_star, varpi, varpi_star, thetaStar,slice_u] = sampleDPM(newJ,newNj, slice_u_star, varpi, varpi_star, alpha, H, M, G,thetaStar,slice_u,alphaa, alphab)
D = 0;
kBar = length(newNj);
while (slice_u_star<=varpi_star) %Step 2
    D = D+1;
    varpi(kBar+D) = varpi_star*betarnd(1,alpha);
    newNj(kBar+D) = 0;
    varpi_star = varpi_star-varpi(kBar+D);
    %对新component抽取参数mu和Sigma
    thetaStar(kBar+D).Sigma = iwishrnd(H.Psi0,H.nv0);%inverse-wishart distribution
    temp = (thetaStar(kBar+D).Sigma)./H.k0;
    thetaStar(kBar+D).mu = mvnrnd((H.mu0)',(temp+temp')./2)';%normal distribution  
    thetaStar(kBar+D).invSigma = inv(thetaStar(kBar+D).Sigma);
    thetaStar(kBar+D).detSigma = det(thetaStar(kBar+D).Sigma);
    %det(thetaStar(kBar+D).Sigma)
end

[thetaStar, newJ, newNj] = Sampling_Components(G, thetaStar, newJ, newNj, varpi, slice_u, M, kBar, D);  % Step 3 

k = kBar+D;
while k >= 1 % delete empty components
        if newNj(k) == 0
            newNj(k) =[];
            thetaStar(k)= [];
            newJ(newJ>k) = newJ(newJ>k) - 1;
        end 
        k=k-1;
end

%disp(newNj')
%alpha = randconparam(alpha, M, length(newNj), alphaa, alphab); 

thetaStar = remix(G, newJ, H);  % Step 4
  
temp_varpi = gamrnd([newNj,alpha], ones(1,length(newNj)+1));%
temp_varpi = temp_varpi./sum(temp_varpi);
varpi = temp_varpi(1:end-1);
varpi_star = temp_varpi(end);  % Step 5 
    
    
ttemp = varpi(newJ);
    
slice_u = unifrnd(zeros(size(ttemp)),ttemp); 
slice_u_star = min(slice_u);   % Step 1 
 
