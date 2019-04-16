function [lamda,R,t] = update(p,n)
% shape 
lamda = p(1:n,1);

% translation
t = p(n+1:n+3);

% rotation
w = p(n+4:n+6);
R = Rodrigues_formula('Rodrigues',w);