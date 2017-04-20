function [tau] = sampleTau_Modified_CCA(a_tau0, b_tau0, S, X, W, N, D, numview, V, S_addition)
for i = 1:numview
    a_tau = a_tau0 + (N*D(i))/2;
    A = X{i} - W{i}*S-V{i}*S_addition{i};
    b_tau = b_tau0 + 0.5*sum(sum(A.^2,1),2);
    tau(i) = gamrnd(a_tau, 1/b_tau);
end
%%$Ã»¸ÄÍê