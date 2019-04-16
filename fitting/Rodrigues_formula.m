function [varargout] = Rodrigues_formula(subfun,varargin)
[varargout{1:nargout}] = feval(subfun,varargin{:}); 


% Rodrigues rotation formula (eq. 4.1)
function R = Rodrigues(w)
wx = skew_symmetric_cross_product_matrix(w);
theta = norm(w);

if theta ~= 0
    R = eye(3) + wx * sin(theta)/theta + wx^2 * (1-cos(theta))/theta^2;
else
    R = eye(3);
end

function w = inverse_Rodrigues(R)
theta = 2*acos(sqrt(trace(R)+1)/2);
v = [R(3,2) - R(2,3); R(1,3) - R(3,1); R(2,1) - R(1,2)];
if norm(v) ~= 0
    w = theta * v/norm(v);
else
    w = zeros(3,1);
end

function w = skew_symmetric_cross_product_matrix(w)
wx = w(1); wy = w(2); wz = w(3);
w = [0 -wz wy; wz 0 -wx; -wy wx 0];
