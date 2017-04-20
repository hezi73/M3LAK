function [G] =  sampleG_CCA_quick(N,M,G,C,Y,Beta,Lambda,Z,thetaStar,S,epsilon,numclass,Leapfrog,Phi_S)
c=0;
for j = 1:M
    current_q = G(:,j);
    
    q = current_q;
    p = randn(size(q));
    current_p = p;
    
    %Make a half step for momentum at the beginning
     %grad_U_q = getGrad_U_q(tau,W,t,X,C,Y,G,Beta,Lambda,m,M,q,n,S,N);
     p = p-epsilon*getGrad_U_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,S,numclass,j,q,Phi_S)/2;
     %Alternate full steps for position and momentum
     for i = 1:Leapfrog
         %Make a full step for the position
         q = q+epsilon*p;
         %Make a full step for the momentum, except at end of trajectory
         if i~=Leapfrog
             p = p-epsilon*getGrad_U_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,S,numclass,j,q,Phi_S);
         end
     end
     
     %Make a half step for momentum at the end
     p = p-epsilon*getGrad_U_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,S,numclass,j,q,Phi_S)/2;
     
     %Negate momentum at end of trajectory to make the proposal symmetric 
     p = -p;
      %Evaluate potential and kenetic energies at start and end of
      %trajectory 
      
      
      current_U = getU_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,j,current_q,S,numclass,Phi_S);
      current_K = sum(current_p.^2)/2;
      proposed_U = getU_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,j,q,S,numclass,Phi_S);
      proposed_K = sum(p.^2)/2;
      %k1=current_U - proposed_U
      %k2=current_K - proposed_K
      %Accept or reject the state at the end of trajectory, returning
      %either the position at the end of trajectory or the initial position
     %1 < exp(current_U - proposed_U + current_K - proposed_K)
      if unifrnd(0,1) < exp(current_U - proposed_U + current_K - proposed_K)
          G(:,j) = q;
          c=c+1;
      else
          G(:,j) = current_q;
      end
      
end
numg=c;
     %G = G_bar;
     

         
    function [grad_U_q] = getGrad_U_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,S,numclass,j,q,Phi_S)
        G(:,j) = q;
        %Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)];
        Phi_S(j,:) = 1/sqrt(M)*cos(G(:,j)'*S);
        Phi_S(j+M,:) = 1/sqrt(M)*sin(G(:,j)'*S);
        derivative_LogLike = 0;
        for c =1:numclass
        Temp{c} = ((C^2*Y{c}.*(Phi_S'*Beta{c})-C^2-C*Lambda{c}).*Y{c})./Lambda{c};
        derivative_LogLike = derivative_LogLike + S*(Temp{c}.*(-Beta{c}(j)*(Phi_S(j+M,:))'+Beta{c}(j+M)*(Phi_S(j,:))'));%计算似然的导数
        end
        k = Z(j);
        derivative_LogPrior= thetaStar(k).invSigma*(q-thetaStar(k).mu);%计算先验的导数
    
        grad_U_q = derivative_LogLike + derivative_LogPrior;%计算后验的导数


        
        function [U_q] = getU_q(N,M,C,Y,G,Beta,Lambda,Z,thetaStar,j,q,S,numclass,Phi_S)
            G(:,j) = q;
            %Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)];
            Phi_S(j,:) = 1/sqrt(M)*cos(G(:,j)'*S);
            Phi_S(j+M,:) = 1/sqrt(M)*sin(G(:,j)'*S);
            k = Z(j);
            %prior = 0.5*log(det(thetaStar(k).Sigma))+0.5*(q-thetaStar(k).mu)'*(thetaStar(k).invSigma)*(q-thetaStar(k).mu);
            prior = 0.5*(q-thetaStar(k).mu)'*(thetaStar(k).invSigma)*(q-thetaStar(k).mu);
            loglike = 0;
            for c = 1:numclass
            loglike = loglike + 0.5* sum( ( ( C*Y{c}.*(Phi_S'*Beta{c})-C-Lambda{c} ).^2 )./Lambda{c} );
            end
            U_q = prior + loglike;
        
       
  