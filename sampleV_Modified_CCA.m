function [V] = sampleV_Modified_CCA(numview, r_v, S_addition, tau, X, D, W, V, K, S)
for i = 1:numview
    for k = 1:K(i)
        V_variance = (1/(r_v{i}(k)+tau(i)*S_addition{i}(k,:)*S_addition{i}(k,:)'));
        temp = (X{i}(:,:)-V{i}*S_addition{i}(:,:)+V{i}(:,k)*S_addition{i}(k,:)-W{i}*S)*tau(i)*S_addition{i}(k,:)';
        V_mu = V_variance*temp;
        V{i}(:,k) = normrnd(V_mu, sqrt(V_variance)*ones(D(i),1), D(i), 1);%normrnd use 标准差 而不是协方差
    end  
end

