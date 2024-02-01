function [W,obj] = PML_ND(X,Y_z,Y_t,beta,lambda,gamma,delta,enaf)
Y=Y_z-Y_t;
num_dim = size(X,2);

W=(X'*X + enaf*eye(num_dim)) \ (X'*Y_t+eps);
H   = (X'*X + enaf*eye(num_dim)) \ (X'*Y);
H1 = H;

R = pdist2( Y_t'+eps, Y'+eps, 'cosine' );

[S,num_p,num_n]= get_S(X, Y_t);
A = [];
for a = 1:size(S, 2)
    A = [A, gamma .* sum(S{a}, 2)];
end

S1 = get_S1(X, Y,num_p,num_n);
A1 = [];
for a = 1:size(S1, 2)
    A1 = [A1, gamma .* sum(S1{a}, 2)];
end

Lip = sqrt(4* (norm(X'*X)^2 + max(max(A1.^2))));
bk = 1;
bk_1 = 1;

iter = 1; obji = 1;
% while 1
for iter=1:50
    Wi = sqrt(sum(W.*W,2)+eps) ;
    d = 0.5./Wi;
    Da = diag(d);
    
    W=W.*(X'*Y_t./((X'*X*W)+beta.*Da*W+delta*H*R+A.*W+eps));
    
    W_s_k  = H + (bk_1 - 1)/bk * (H - H1);
    Gw_s_k = W_s_k - 1/Lip * (X'*X*W_s_k - X'*Y +  A1 .* W_s_k+delta * W * R );
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    H1  = H;
    H    = softthres(Gw_s_k,lambda/Lip);
    
    sample = 0;
    for a = 1:size(S, 2)
        sample = sample + W(:, a)' * S{a} * W(:, a);
    end
    sample1 = 0;
    for a = 1:size(S, 2)
        sample1 = sample1 + H(:, a)' * S1{a} * H(:, a);
    end
    obj(iter)=norm((X*W-Y_t),'fro')^2+norm((X*H-Y),'fro')^2+lambda*sum(sum(abs(H)), 2)...
        +beta*trace(W'*Da*W)+(gamma*sample+gamma*sample1) / 2+delta*trace(R*W'*H);
    cver = abs((obj(iter) - obji)/obji);
    obji = obj(iter);
    iter = iter + 1;
    if (cver < 10^-3 && iter > 2) , break, end
end
end
function W = softthres(W_t,lambda)
W = max(W_t-lambda,0) - max(-W_t-lambda,0);
end