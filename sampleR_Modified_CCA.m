function [r] = sampleR_Modified_CCA(a_r0, b_r0, W, numview, D, K0)
for i = 1:numview
    for k =1:K0
        a_r = a_r0 + 1*D(i)/2;
        b_r = b_r0 + 0.5*sum(W{i}(:,k).^2,1);
        r(i,k) = gamrnd(a_r,1/b_r);
    end
end