function remixedThetaStar = remix(G, J, H)
    u1 = sort(unique(J));%unique(J)�����ܹ��ж��ٸ��࣬��K��
    %sort�Ǵ�С��������{��ʵunique����������������ܣ���С��������}
    %����u1 = ��1,2,3����������K��
    for i = 1:length(u1)  %�ɲ���ʵ��
        X = G(:,J == i);%ͳ��componentΪi������g�ļ��ϣ��Ǹ�����
        numData = size(X,2);%���ڵ�i��component�����ݵĸ���

        
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
%         clear X;%Xÿ�εĴ�С��һ��
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
