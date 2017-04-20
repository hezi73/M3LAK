function [S_addition] = sampleS_addition_Modified_CCA_BNK(N, numview, K, S, V, tau, X, W, covriance_S_addition)
for n = 1:N
    for i = 1:numview
        temp =  V{i}'*V{i}*tau(i);
        temp1 = V{i}'*(X{i}(:,n)-W{i}*S(:,n))*tau(i);
        Sigma_SNv = inv(covriance_S_addition*eye(K(i))+temp);
        mu_SNv = Sigma_SNv*temp1;
        S_addition{i}(:,n) = mvnrnd(mu_SNv',(Sigma_SNv+Sigma_SNv')./2)';
    end
end