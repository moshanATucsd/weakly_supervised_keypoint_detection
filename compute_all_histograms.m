function [] = compute_all_histograms()

clear;clc;close all;

%%

run /Users/moshan/Documents/Research/deep_learning/matconvnet-1.0-beta23/matlab/vl_setupnn;
addpath('MATLAB');

%%

addpath('vis');
addpath('fitting');
addpath('common');

%%
rootDir_data = '/Users/moshan/Documents/Research/cse291_class_project/datasets/FG3DCar';
% rootDir_data = '/data3/moshan/cse291/class_proj/FG3DCar';

% load face
load(fullfile(rootDir_data,'/3dmodel/car_face_mesh.mat'));
face = carFace_mesh;
% load landmark
landmark = textread(fullfile(rootDir_data,'/3dmodel/landmark_ext.txt'),'%f');
num_landmark = numel(landmark);
% load ground truth
load(fullfile(rootDir_data,'/3dmodel/ground_truth'),'manualParam');

% get image list
car = 'original/';
imgDir = fullfile(rootDir_data, '/dataset/',car);
images = dir(fullfile(rootDir_data, '/dataset/',car,'*.jpg'));
images = cellfun(@(x) x,{images.name},'UniformOutput',false);

%%

rootDir_net = '/Users/moshan/Documents/Research/cse291_class_project';
net = load(fullfile(rootDir_net,'/models/imagenet-vgg-verydeep-19.mat'));

% net = load('/data3/moshan/cse291/class_proj/imagenet-vgg-verydeep-19.mat');

%%
heatmapsDir = 'heatmaps/';
heatmapsDir = fullfile(rootDir_data, '/dataset/',heatmapsDir);

histDir = 'histograms/';
histDir = fullfile(rootDir_data, '/dataset/',histDir);

% loop over each image
for i=1:length(images)
    
    %%
    % find corresponding index in ground truth file
    index = find(cellfun(@(x) strcmp(x.filename,images(i)),manualParam));
    
    % load image
    filename = manualParam{index}.filename;
    img = imread(fullfile(imgDir,filename));
    matName = fullfile(heatmapsDir,strcat(filename(1:end-3),'mat'));
    histName = fullfile(histDir,strcat(filename(1:end-4),'_hist.mat'));
    
    fprintf('Processing %s\n', filename);
    % load 2D points
    points(:,1:2) = manualParam{index}.bestModel(:,1:2);
    
    % load visible faces
    visible_faces = manualParam{index}.visibleFace;
    
    % visible points
    visible_point = cal_visible_point(visible_faces,face,256);
    
    %     imshow(img); hold on;
    color = 'y';
    [kp_visible, landmark_visible] = visualize_landmark(points,visible_point,landmark,color);
    box = kp2box(img, kp_visible);
    %%
    [keypoints, scores] = kp_detection_pr(net,img,box,matName);
    
    %% PCK
    
    [detected_kps, detected_kp_scores, kp_label, detected_gt_kp] = compare_kp_pr(box, keypoints, scores, kp_visible, landmark_visible);
    
    landmark_histogram = zeros(1,num_landmark);
    for j = 1:numel(kp_label)
        landmark_histogram(landmark==kp_label(j)) = landmark_histogram(landmark==kp_label(j))+1;
    end
    
    save(histName, 'landmark_histogram');
end

end

function [corners_sub, scores] = kp_detection_pr(net,im,box,matName)
%weakly supervised keypoint detection

%%

im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
mean_image = ones(size(im_));
mean_values = net.meta.normalization.averageImage;
for i = 1:3
    mean_image(:,:,i) = mean_values(i)*mean_image(:,:,i);
end
im_ = im_ - mean_image;

box_small = box;
[h,w,~] = size(im);
[new_h,new_w,~] = size(im_);
scaleH = new_h/h;
scaleW = new_w/w;
box_small(1) = box_small(1)*scaleW;
box_small(3) = box_small(3)*scaleW;
box_small(2) = box_small(2)*scaleH;
box_small(4) = box_small(4)*scaleH;
% figure;
% imshow(im_); hold on;
% rectangle('Position',box_small,'EdgeColor','r');
%%

%save and load results to increase efficiency
load(matName);

coarse_map = mask_occlusion;%low resolution
fine_map = mask_deconv;%high resolution

feat_com = 1./(1+exp(-coarse_map - fine_map));

%%

%non maxima suppression
nms_window_size = 30;
nms_img = non_maxima_suppression( feat_com, nms_window_size);

[h,w] = size(nms_img);
box_mask = zeros(h,w);
box = round(box);
box_mask(max(1,box(2)):min(box(2)+box(4),h),max(1,box(1)):min(box(1)+box(3),w)) = 1;
nms_img = nms_img.*box_mask;

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
[corners, corners_sub, scores]=detect_subpixel_corners(nms_img,sub_window_size,...
    Ix,Iy,Ix2,IxIy,Iy2);
end

function [detected_kp, detected_kp_scores, kp_label, detected_gt_kp] = compare_kp_pr(box, keypoints, scores, kp_visible, landmark_visible)

detected_kp = [];
kp_label = [];
detected_gt_kp = [];
detected_kp_scores = [];

alpha = 0.2;%0.1 or 0.2 in paper
radius = alpha*(max(box(3), box(4)));

num_kp = size(keypoints,1);
num_landmark = size(kp_visible,1);

for i = 1:num_kp
    kp = keypoints(i,:);
    kps = repmat(kp,num_landmark,1);
    distances = diag((kps-kp_visible)*(kps-kp_visible)');
    [min_dist, min_ind] = min(distances);
    if min_dist < radius
        detected_kp = [detected_kp;kp];
        detected_kp_scores = [detected_kp_scores;scores(i)];
        kp_label = [kp_label;landmark_visible(min_ind)];
        detected_gt_kp = [detected_gt_kp;kp_visible(min_ind,:)];
    end
end

end