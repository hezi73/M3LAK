function [thetaStar, J, nj] = Sampling_Components(G, thetaStar, J, nj, varpi, slice_u, M, kBar, D) 
    oldJ = zeros(size(J));
    for i = 1:M   %可并行实现
        oldJ(i) = J(i);
        q1 = zeros(kBar, 1);
        for k = 1:kBar
            if varpi(k)>=slice_u(i)               
                q1(k) = GetLogLike(G(:, i), thetaStar(k).mu, thetaStar(k).invSigma, thetaStar(k).detSigma);
            else
                q1(k) = -inf;  
            end
        end
        q2 = zeros(D, 1);
        for k = 1:D
            if varpi(kBar+k)>=slice_u(i)
                q2(k) = GetLogLike(G(:, i), thetaStar(kBar+k).mu, thetaStar(kBar+k).invSigma, thetaStar(kBar+k).detSigma);  
            else
                q2(k) = -inf;  
            end
        end
        q = [q1; q2];
        qMax = max(q);
        qRel = q - qMax;
        q = exp(qRel);
        q = q./sum(q);
        qCumSum = cumsum(q);
        u = rand;
        temp = find (qCumSum >= u); 
%         if numel(temp)==0
%         picked(i)=1;
%         J(i) = picked(i);
%         else
        picked(i) = temp(1);
        J(i) = picked(i);   
%         end
    end    
    for i = 1:M
        if picked(i)~=oldJ(i)
            nj(oldJ(i)) = nj(oldJ(i)) - 1;
            nj(picked(i)) = nj(picked(i)) + 1;  
        end
    end
    
function [h] = GetLogLike(g, mu, invSigma, detSigma)%g服从normal distribution（mu,Sigma） g是列向量
    %h = (-0.5)*log(2*pi*abs(det(Sigma)))-0.5*(g-mu)'*Sigma*(g-mu);
%     Sigma = Sigma + .001;
%     diffX = (g - mu);
%     logLikeX = sum(-log(Sigma) - ((diffX).^2)./(2*(Sigma.^2))); 
%     h = logLikeX;
h = (-0.5)*length(g)*log(2*pi)+(-0.5)*log(detSigma)+(-0.5)*(g-mu)'*invSigma*(g-mu);

    