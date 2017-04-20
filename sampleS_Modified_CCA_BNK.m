function [S] = sampleS_Modified_CCA_BNK(tau,W,V,X,C,Y,Beta,Lambda,K0,M,S,N,numview,S_addition, numclass, epsilon, Leapfrog, G)
c=0;
for n = 1:N %size(A,1)返回行数；size(A,2)返回列数，这里应该是N
    current_q = S(:,n);
    
    q = current_q;
    p = randn(size(q));
    current_p = p;
    
    %Make a half step for momentum at the beginning
     %grad_U_q = getGrad_U_q(tau,W,t,X,C,Y,G,Beta,Lambda,m,M,q,n,S,N);
     p = p-epsilon*getGrad_U_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition, numclass,q)/2;
     %Alternate full steps for position and momentum
     for i = 1:Leapfrog
         %Make a full step for the position
         q = q+epsilon*p;
         %Make a full step for the momentum, except at end of trajectory
         if i~=Leapfrog
             p = p-epsilon*getGrad_U_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition, numclass,q);
         end
     end
     
     %Make a half step for momentum at the end
     p = p-epsilon*getGrad_U_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition, numclass,q)/2;
     
     %Negate momentum at end of trajectory to make the proposal symmetric 
     p = -p;
      %Evaluate potential and kenetic energies at start and end of
      %trajectory 
      current_U = getU_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition,numclass,current_q);
      current_K = sum(current_p.^2)/2;
      proposed_U = getU_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition,numclass,q);
      proposed_K = sum(p.^2)/2;
      
      %Accept or reject the state at the end of trajectory, returning
      %either the position at the end of trajectory or the initial position
      %accepable = exp(current_U - proposed_U + current_K - proposed_K)
      
      if unifrnd(0,1) < exp(current_U - proposed_U + current_K - proposed_K)
          S(:,n) = q;
          c=c+1;
      else
          S(:,n) = current_q;
      end    
end
nums=c;

    function grad_U_q = getGrad_U_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition, numclass,q)
        %S(:,n) = q;
        %Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)];
        Phi_S_n = [1/sqrt(M)*[cos(G'*q);sin(G'*q)];1];
        temp_a = Phi_S_n(M+1:2*M,1);
        temp_b = Phi_S_n(1:M,1);
        derivative_phi_q = [-repmat((temp_a)',K0,1).*G repmat((temp_b)',K0,1).*G zeros(K0,1)];
        derivative_Loglike_q_1 = 0;
        for c = 1 :numclass
        derivative_Loglike_q_1 = derivative_Loglike_q_1 + (C+Lambda{c}(n)-C*Y{c}(n)*Phi_S_n'*Beta{c})*(Lambda{c}(n)^(-1))*derivative_phi_q*(-C*Y{c}(n)*Beta{c});
        end
        derivative_Loglike_q_2 = 0;
        for i = 1:numview
        derivative_Loglike_q_2 = derivative_Loglike_q_2 + tau(i)*W{i}'*(W{i}*q+V{i}*S_addition{i}(:,n)-X{i}(:,n));
        end
        derivative_Loglike_q = derivative_Loglike_q_1 + derivative_Loglike_q_2;
        derivative_Prior_q = q;
        grad_U_q = derivative_Loglike_q  + derivative_Prior_q;
        
        function U_q = getU_q(tau,W,X,C,Y,G,Beta,Lambda,K0,M,n,S,V,numview,S_addition,numclass,q)
            %S(:,n) = q;
            %Phi_S = [1/sqrt(M)*[cos(G'*S);sin(G'*S)];ones(1,N)];
            Phi_S_n = [1/sqrt(M)*[cos(G'*q);sin(G'*q)];1];
            prior = 0.5*q'*eye(K0)*q;
            loglike1 = 0;
            for c = 1:numclass
                loglike1 = loglike1 + 0.5*(C+Lambda{c}(n)-C*Y{c}(n)*Phi_S_n'*Beta{c})^2/Lambda{c}(n);
            end
            loglike2 = 0;
            for i = 1:numview
                loglike2 = loglike2 + 0.5*(X{i}(:,n)-W{i}*q-V{i}*S_addition{i}(:,n))'*tau(i)*(X{i}(:,n)-W{i}*q-V{i}*S_addition{i}(:,n)) ;
            end
            U_q = prior + loglike1 + loglike2;