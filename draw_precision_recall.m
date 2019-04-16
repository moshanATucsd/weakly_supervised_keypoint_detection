function [] = draw_precision_recall()

clear;clc;close all;

%%

% run /Users/moshan/Documents/Research/deep_learning/matconvnet-1.0-beta23/matlab/vl_setupnn;
% addpath('MATLAB');

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
resultsDir = 'precision_recall/';
heatmapsDir = 'heatmaps/';
heatmapsDir = fullfile(rootDir_data, '/dataset/',heatmapsDir);

% loop over each image
percentages = 0.05:0.01:1;
pr_num = numel(percentages);
pr_values = zeros(pr_num,2);
% for i=1:length(images)
for i=1:10
    
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
    
    %     imshow(img); hold on;
    color = 'y';
    [kp_visible, landmark_visible] = visualize_landmark(points,visible_point,landmark,color);
    box = kp2box(img, kp_visible);
    %%
    [keypoints, scores] = kp_detection_pr(net,img,box,matName);
    
    %% PCK
    
    [detected_kps, detected_kp_scores, kp_label, detected_gt_kp] = compare_kp_pr(box, keypoints, scores, kp_visible, landmark_visible);
    
    %% precision-recall
    %soft thresholding
    total_kp_num = numel(scores);
    [scores_sorted, scores_index] = sort(scores,'descend');
    
    for j = 1:pr_num
        
        kp_num_to_keep = ceil(total_kp_num*percentages(j));
        threshold = scores_sorted(kp_num_to_keep);
        
        thresholded_index = scores > threshold;
        thresholded_kps = keypoints(thresholded_index,:);
        
        thresh_ind_detected = detected_kp_scores > threshold;
        thresh_kps_detected = detected_kps(thresh_ind_detected,:);
        
        true_positive = size(thresh_kps_detected,1);
        precision = true_positive / size(thresholded_kps,1);
        
        recall = true_positive / size(kp_visible,1);
        
        pr_values(j,1) = pr_values(j,1) + precision;
        pr_values(j,2) = pr_values(j,2) + recall;
    end
    
end

pr_values = pr_values/length(images);

save('pr_values.mat','pr_values');

figure;
plot(pr_values(:,2),pr_values(:,1));

%% save images

iptsetpref('ImshowBorder','tight');
set(gca,'LooseInset',get(gca,'TightInset'))
saveas(gca,fullfile(imgDir,resultsDir,filename))
close all;

end

function [corners_sub, scores] = kp_detection_pr(net,im,box,matName)
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