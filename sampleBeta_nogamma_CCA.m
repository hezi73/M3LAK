function [Beta,Sigma_beta_gamma,temp_sigma_beta] = sampleBeta_nogamma_CCA(v,C,Lambda,Phi_S,Y,M,numclass)
for c = 1:numclass
%calculate Sigma_beta first
temp_sigma_beta = v{c}*eye(2*M+1)+C^2*Phi_S*diag(1./Lambda{c})*Phi_S';
Sigma_beta_gamma = inv(temp_sigma_beta);%+0.0001*eye(2*M+1)
%calculate mu_beta second
temp_mu_beta = Phi_S*((C^2./Lambda{c}+C).*Y{c});
mu_beta_gamma = Sigma_beta_gamma*(temp_mu_beta);
%sample Beta from Normal distribution with parameter mu_beta & Sigma_beta
Beta_gamma = mvnrnd(mu_beta_gamma',(Sigma_beta_gamma+Sigma_beta_gamma')./2)';
%sigma_beta is variance || mvnrnd nedd covariance || the difference
Beta{c} = Beta_gamma;
end
