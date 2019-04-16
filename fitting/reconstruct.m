function X = reconstruct(G,U,lamda)
X = zeros(3,size(U,1)/3); 
X(:) = G(:,1:10)*lamda(1:10,1) + U;