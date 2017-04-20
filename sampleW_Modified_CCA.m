function [W] = sampleW_Modified_CCA(numview, r, S_addition, tau, X, D, W, V, K0, S)
for i = 1:numview
    for k = 1:K0
        W_variance = (1/(r(i,k)+tau(i)*S(k,:)*S(k,:)'));
        temp = (X{i}(:,:)-W{i}*S(:,:)+W{i}(:,k)*S(k,:)-V{i}*S_addition{i})*tau(i)*S(k,:)';
        W_mu = W_variance*temp;
        W{i}(:,k) = normrnd(W_mu, sqrt(W_variance)*ones(D(i),1), D(i), 1);%normrnd use 标准差 而不是协方差
    end  
end
