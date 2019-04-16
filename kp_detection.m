function [ corners_sub ] = kp_detection(net,im,box)
%weakly supervised keypoint detection

%%

im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

box_small = box;
[h,w,~] = size(im);
[new_h,new_w,~] = size(im_);
scaleH = new_h/h;
scaleW = new_w/w;
box_small(1) = box_small(1)*scaleW;
box_small(3) = box_small(3)*scaleW;
box_small(2) = box_small(2)*scaleH;
box_small(4) = box_small(4)*scaleH;
figure;
imshow(im_); hold on;
rectangle('Position',box_small,'EdgeColor','r');
%%

[heatmap_deconv, mask_deconv] = graphical_deconvolution(net, im, im_, box_small);

[heatmap_occlusion, mask_occlusion, overlay] = graphical_occlusion(net, im, im_,box_small,{16, 8});

coarse_map = mask_occlusion;%low resolution
fine_map = mask_deconv;%high resolution

feat_com = 1./(1+exp(-coarse_map - fine_map));

%%

%non maxima suppression
nms_window_size = 30;
nms_img = non_maxima_suppression( feat_com, nms_window_size);

[h,w] = size(nms_img);
box_mask = zeros(h,w);
box_mask(box(2):box(2)+box(4),box(1):box(1)+box(3)) = 1;
nms_img = nms_img.*box_mask;

% kp_num = round(nnz(nms_img)/2);
% % kp_num = 15;
% [sorted, index] = sort(nms_img(:));
% threshold = sorted(end-kp_num);
kp_map_bw = nms_img;
% kp_map_bw(kp_map_bw<threshold) = 0;
% kp_map_bw(kp_map_bw~=0) = 1;

%%

image = rgb2gray(im);
[Ix, Iy] = compute_gradient(image);

sub_window_size = 5;
kernel_sum = ones(sub_window_size, sub_window_size);

%use conv2
Ix2 = conv2(Ix.^2, kernel_sum, 'same');
IxIy = conv2(Ix.*Iy, kernel_sum, 'same');
Iy2 = conv2(Iy.^2, kernel_sum, 'same');
%detect corners with subpixel accuracy
[corners, corners_sub, scores]=detect_subpixel_corners(kp_map_bw,sub_window_size,...
    Ix,Iy,Ix2,IxIy,Iy2);

end

