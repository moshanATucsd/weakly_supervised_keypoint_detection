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

% load face
load(fullfile(rootDir_data,'/3dmodel/car_face_mesh.mat'));
face = carFace_mesh;
% load landmark
landmark = textread(fullfile(rootDir_data,'/3dmodel/landmark_ext.txt'),'%f');
% load ground truth
load(fullfile(rootDir_data,'/3DModel/ground_truth'),'manualParam');

% get image list
car = 'original/';
imgDir = fullfile(rootDir_data, '/dataset/',car);
images = dir(fullfile(rootDir_data, '/dataset/',car,'*.jpg'));
images = cellfun(@(x) x,{images.name},'UniformOutput',false);

%%

rootDir_net = '/Users/moshan/Documents/Research/cse291_class_project';
net = load(fullfile(rootDir_net,'/models/imagenet-vgg-verydeep-19.mat'));

%%
resultsDir = 'qualitative_results_all/';
heatmapsDir = 'heatmaps/';
heatmapsDir = fullfile(rootDir_data, '/dataset/',heatmapsDir);

% loop over each image
for i=1:length(images)
    
    %%
    resultsDir1 = fullfile(resultsDir,num2str(i));
    resultsDir2 = fullfile(rootDir_data,resultsDir1);
    if ~exist(resultsDir2, 'dir')
        mkdir(resultsDir2);
    end
    
    %%
    % find corresponding index in ground truth file
    index = find(cellfun(@(x) strcmp(x.filename,images(i)),manualParam));
    
    % load image
    filename = manualParam{index}.filename;
    img = imread(fullfile(imgDir,filename));
    matName = fullfile(heatmapsDir,strcat(filename(1:end-3),'mat'));
    fprintf('Processing %s\n', filename);
    % load 2D points
    points(:,1:2) = manualParam{index}.bestModel(:,1:2);
    
    % load visible faces
    visible_faces = manualParam{index}.visibleFace;
    
    % visible points
    visible_point = cal_visible_point(visible_faces,face,256);
    
    imshow(img); hold on;
    color = 'y';
    [kp_visible, landmark_visible] = visualize_landmark(points,visible_point,landmark,color);
    box = kp2box(img, kp_visible);
    
    %%
    [keypoints, heatmap_deconv, overlay] = kp_detection_qualitative(net,img,box,matName);
    
    %% PCK
    
    [detected_kp, kp_label, detected_gt_kp] = compare_kp(box, keypoints, kp_visible, landmark_visible);
    
    %% Plot and save
    close all;
    
    I1 = imcrop(img,box);
    imwrite(I1,fullfile(resultsDir2,'img.png'));
    
    I2 = imcrop(heatmap_deconv,box);
    imwrite(I2,fullfile(resultsDir2,'saliency.png'));
    
    I3 = imcrop(overlay,box);
    imwrite(I3,fullfile(resultsDir2,'heatmap.png'));
    
    figure;
    imshow(I1);
    hold on
    keypoints = keypoints - box(1:2);
    scatter(keypoints(:,1),keypoints(:,2),100,'filled','r');
    
    iptsetpref('ImshowBorder','tight');
    set(gca,'LooseInset',get(gca,'TightInset'));
    saveas(gca,fullfile(resultsDir2,'detected_kp.png'));
    close all;
    
    figure;
    imshow(I1);
    hold on
    kp_visible = kp_visible - box(1:2);
    scatter(kp_visible(:,1),kp_visible(:,2),100,'filled','g');
    
    iptsetpref('ImshowBorder','tight');
    set(gca,'LooseInset',get(gca,'TightInset'));
    saveas(gca,fullfile(resultsDir2,'gt.png'));
    close all;
end

function [ corners_sub, heatmap_deconv, overlay_occlusion ] = kp_detection_qualitative(net,im,box,matName)
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

%soft thresholding
total_kp_num = numel(scores);
percentage = 0.4;
kp_num_to_keep = round(total_kp_num*percentage);
[sorted, index] = sort(scores,'descend');
scores = sorted;
corners = corners(index,:);
corners_sub = corners_sub(index,:);

corners_sub = corners_sub(1:kp_num_to_keep,:);
end

