function p = initialize(lamda,R,t,n)
% shape
p(1:n,1) = lamda;

%translation
p(n+1:n+3) = t;

% rotation
w = Rodrigues_formula('inverse_Rodrigues',R);
p(n+4:n+6) = w;