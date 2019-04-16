clear;clc;close all;

%%
% run /Users/moshan/Documents/Research/deep_learning/matconvnet-1.0-beta23/matlab/vl_setupnn;
% addpath('MATLAB');

% net = load('imagenet-caffe-alex.mat');
net = load('/Users/moshan/Documents/Research/cse291_class_project/models/imagenet-vgg-verydeep-19.mat');
% net = DDagNN.loadobj(load('/Users/moshan/Documents/Research/cse291_class_project/models/imagenet-resnet-50-dag.mat'));

%%
im = imread('Acura_ZDX_2010_01.jpg');

im = imresize(im, .5);

im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = im_ - net.meta.normalization.averageImage;

%%

[heatmap_deconv, mask_deconv] = graphical_deconvolution(net, im, im_);
figure;
imshow(heatmap_deconv);
title('deconv heatmap')

%%

[heatmap_occlusion, mask_occlusion] = graphical_occlusion(net, im, im_, {8, 8});
figure;
imshow(heatmap_occlusion);
title('occlusion heatmap')

% lambda = 0.8; %relative weight of the image gradients

feat6 = mask_occlusion;%low resolution
feat12 = mask_deconv;%high resolution

featConv6 = 1./(1+exp(-feat6));
featConv12 = 1./(1+exp(-feat12));
featConv = 1./(1+exp(-feat6 - feat12));
figure;
imshow(featConv);
title('final heatmap')

%%
%non maxima suppression
nms_window_size = 30;
nms_img = non_maxima_suppression( featConv, nms_window_size);
% kp_num = round(nnz(nms_img)/2);
kp_num = 15;
[sorted, index] = sort(nms_img(:));
threshold = sorted(end-kp_num);
kp_map_bw = nms_img;
kp_map_bw(kp_map_bw<threshold) = 0;
kp_map_bw(kp_map_bw~=0) = 1;
% figure;
% imshow(mat2gray(kp_map_bw));
% title('Kp map')
% 
% [row_ind, col_ind] = find(kp_map_bw);
% figure;
% imshow(im); hold on
% scatter(col_ind,row_ind,100,'filled','g');
% title('Kp result')

%%
image = rgb2gray(im);
[Ix, Iy] = compute_gradient(image);
%sanity check
% figure;
% subplot(1,2,1);
% imshow(mat2gray(Ix));
% subplot(1,2,2);
% imshow(mat2gray(Iy));

sub_window_size = 5;
kernel_sum = ones(sub_window_size, sub_window_size);

%use conv2
Ix2 = conv2(Ix.^2, kernel_sum, 'same');
IxIy = conv2(Ix.*Iy, kernel_sum, 'same');
Iy2 = conv2(Iy.^2, kernel_sum, 'same');
%detect corners with subpixel accuracy
[corners, corners_sub]=detect_subpixel_corners(kp_map_bw,sub_window_size,...
    Ix,Iy,Ix2,IxIy,Iy2);
%plot subpixel feature
figure;
imshow(image);
hold on
plot(corners_sub(:,1), corners_sub(:,2), 'r*');
plot(corners(:,1), corners(:,2), 'bo');

figure;
imshow(im); hold on
scatter(corners_sub(:,1), corners_sub(:,2),100,'filled','r');
title('Kp result')