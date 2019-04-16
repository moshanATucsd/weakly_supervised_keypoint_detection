function [ Ix, Iy ] = compute_gradient ( image )
%five point central difference operator
kernel_x = [-1,8,0,-8,1] / 12;
kernel_y = kernel_x';

Ix = conv2(double(image), kernel_x, 'same');
Iy = conv2(double(image), kernel_y, 'same');

end

