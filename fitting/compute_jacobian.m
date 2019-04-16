%{ 
  compute jacobian matrix
    Input:
        u: 1x2, landmark point
        v: 1x2, corresponding point
       ua: 1x2, end point of boundary segment
       ub: 1x2, end point of boundary segment
        k: vertex index
        P: 3x4, projection matrix
        G: dxn, span by eigen-vectors
        U: dx1, mean vector
        p: Nx1, parameter matrix
     type: optimization type
   Output:
       ji: 1xN,Jacobian matrix
       ei: 1x1,signed distance error
%}
function [ji,ei] = compute_jacobian(u,v,ua,ub,k,P,G,U,p,type)
n = size(G,2);

% compute j3
[j3,xi] = get_j3(p,U,G,k,type);

% conver from world to vehicle coordinate
w = p(n+4:n+6);
t = p(n+1:n+3);
R = Rodrigues_formula('Rodrigues',w);
xi_veh = R * xi + t;

% compute j2
j2 = get_j2(P,xi_veh);

% compute j1
[j1,e] = get_j1(u,v,ua,ub);

ji = j1*j2*j3;
ei = e;


%% jacobian 
%--------------------------------------------------------------------------
%{  
    3d Jacobian
    Input
        p	: Nx1 paramter matrix
        U	: dx1 matrix, mean vector
        G	: dxn matrix, span by eignevectors
        k	: vertex index
        type: optimization type
    Return
        j3  : 3-d Jacobian, 3xN 
        xi  : vehicle coordinate
%}
function [j3,xi] = get_j3(p,U,G,k,type)
n = size(G,2);
N = size(p,1);
j3 = zeros(3,N);

%js
Gk = G(3*(k-1)+1:3*(k-1)+3,:);
js = Gk;

%jt
jt = eye(3);

%jr
ps = p(1:n);
uk = U(3*(k-1)+1:3*(k-1)+3,:);
xi = Gk * ps + uk;
xix = Rodrigues_formula('skew_symmetric_cross_product_matrix',xi);
jr = xix';

switch type
    case {'shape'}
        j3(:,1:n) = js;
    case {'pose'}
        j3(:,n+1:n+3) = jt;
        j3(:,n+4:n+6) = jr;
    case {'all'}
        j3(:,1:n) = js;
        j3(:,n+1:n+3) = jt;
        j3(:,n+4:n+6) = jr;
    otherwise
        disp('wrong type!');
end
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%{
	projection Jacobian
    Input
        P: 3x4 projection matrix
    Output
        j2: 2x3 Jacobian of Ei2
%}
function j2 = get_j2(P,xi)
A1 = cal_Ak(P,1);
A2 = cal_Ak(P,2);
b1 = cal_bk(P,1);
b2 = cal_bk(P,2);
m3 = P(3,1:3)';
t3 = P(3,4);

j2 = [xi'*A1+b1';
      xi'*A2+b2']/(m3'*xi+t3)^2;
%--------------------------------------------------------------------------
 


%--------------------------------------------------------------------------
%{
    distance Jacobian (modified)
    Input
      u: landmark point
      v: corresoning point
      ua,ub: end points of line segment
    Output
      j1: 1x2 Jacobian of Ei1
      e: signed distance error
 %}
      
function [j1,e] = get_j1(u,v,ua,ub)

% line normal
N = [0,-1;1,0];
w = (ua - ub)';
n = N * w;
n = n/norm(n);

% theta
d = (v-u)';
theta = atan2(d(2),d(1))-atan2(n(2),n(1));
R = [cos(theta),-sin(theta);sin(theta),cos(theta)];

% rotated normal
n = (R * n);
n = n/norm(n);

% signed distance error
e = -(n'* d);
j1 = -n';
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function Ak = cal_Ak(P,k)
m3 = P(3,1:3)';
mk = P(k,1:3)';
Ak = m3 * mk' - mk * m3';
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function bk = cal_bk(P,k)
mk = P(k,1:3)';
t3 = P(3,4);
m3 = P(3,1:3)';
tk = P(k,4);
bk = mk * t3 - m3 * tk;
%--------------------------------------------------------------------------







