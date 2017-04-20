function [Lambda] = sampleLambda_CCA(N,C,Phi_S,Beta,Y,numclass)
%Sample Lambda from inverse Gaussian IG(1./|C*(1-Y.*(Phi_S'*Beta))|,1)
for c = 1:numclass
mean = 1./abs(C*(1-Y{c}.*(Phi_S'*Beta{c})));
    
v = normrnd(0,1,[N,1]);
y = v.^2;
x = mean+((mean.^2).*y)./2-(mean./2).*sqrt((4.*mean).*y+(mean.^2).*(y.^2));
z = rand(N,1);
    
temp = mean./(mean+x)-z;
    
newLambda = x;
I = find(temp<0);
newLambda(I) = (mean(I).^2)./x(I);
    
newLambda = 1./max(eps,newLambda);

Lambda{c} = newLambda;
end