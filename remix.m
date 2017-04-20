function remixedThetaStar = remix(G, J, H)
    u1 = sort(unique(J));%unique(J)代表总共有多少个类，既K；
    %sort是从小到大排序{其实unique本身好像就有这个功能，从小到大排序}
    %最后的u1 = 【1,2,3，。。。，K】
    for i = 1:length(u1)  %可并行实现
        X = G(:,J == i);%统计component为i的所有g的集合，是个矩阵
        numData = size(X,2);%属于第i个component的数据的个数

        
        % Normal inverse Wishart prior
        % see Bayesian Nonparametric Kernel learning     
        meanX = mean(X,2);
        Psi = H.Psi0+(X-repmat(meanX,1,numData))*(X-repmat(meanX,1,numData))'+(H.k0*numData/(H.k0+numData))*(meanX-H.mu0)*(meanX-H.mu0)';
        nv = H.nv0+numData;
        Sigma = iwishrnd((Psi+Psi')./2,nv);%+eps*eye(size(Psi,1))
        tempmu = (H.k0*H.mu0+numData*meanX)./(H.k0+numData);
        tempSigma = Sigma/(H.k0+numData);
        mu = mvnrnd(tempmu',tempSigma)';
        
        remixedThetaStar(i).mu = mu;
        remixedThetaStar(i).Sigma = Sigma;
        remixedThetaStar(i).invSigma = inv(Sigma);
        remixedThetaStar(i).detSigma = det(Sigma);
%         clear X;%X每次的大小不一样
%         clear Psi;
%         clear Sigma;
%         clear mu;
%         clear tempmu;
%         clear tempSigma;
          X = [];
          Psi = [];
          Sigma = [];
          mu = [];
          tempmu = [];
          tempSigma = [];
        
    end
