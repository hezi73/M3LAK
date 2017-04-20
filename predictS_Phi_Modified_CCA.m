function S_test = predictS_Phi_Modified_CCA(tau,W,V,xtest,K0,S_test,Ntest,numview)
for nv = 1:numview
    inv_Phi{nv} = inv(V{nv}*V{nv}'+tau(nv)^(-1)*eye(size(V{nv},1)));
end
for n = 1:Ntest
    temp_sigma = 0;
    temp_mu = 0;
    for i = 1:numview
        temp_sigma = temp_sigma + W{i}'*inv_Phi{i}*W{i};
        temp_mu = temp_mu + (xtest{i}(:,n)'*inv_Phi{i}*W{i})';
    end
    Sigma_S =  inv(eye(K0) + temp_sigma);
    mu_S = Sigma_S*(temp_mu);
    S_test(:,n) = mvnrnd(mu_S',(Sigma_S + Sigma_S')./2)';
end