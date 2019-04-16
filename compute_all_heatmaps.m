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
load(fullfile(rootDir_data,'/3dmodel/ground_truth.mat'),'manualParam');

% get image list
car = 'original/';
imgDir = fullfile(rootDir_data, '/dataset/',car);
images = dir(fullfile(rootDir_data, '/dataset/',car,'*.jpg'));
images = cellfun(@(x) x,{images.name},'UniformOutput',false);

%%

rootDir_net = '/Users/moshan/Documents/Research/cse291_class_project';
net = load(fullfile(rootDir_net,'/models/imagenet-vgg-verydeep-19.mat'));

%%
heatmapsDir =fullfile(rootDir_data, '/dataset/heatmaps/');

% loop over each image
for i=300-25:length(images)
    %%
    
    % find corresponding index in ground truth file
    index = find(cellfun(@(x) strcmp(x.filename,images(i)),manualParam));
    
    % load image
    filename = manualParam{index}.filename;
    matName = fullfile(heatmapsDir,strcat(filename(1:end-3),'mat'));
    img = imread(fullfile(imgDir,filename));
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
    save_heatmaps(net,img,box,matName);
    
    
end

function [] = save_heatmaps(net,im,box,matName)
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
figure;
imshow(im_); hold on;
rectangle('Position',box_small,'EdgeColor','r');
%%

%save and load results to increase efficiency
[heatmap_deconv, mask_deconv] = graphical_deconvolution(net, im, im_, box_small);
[heatmap_occlusion, mask_occlusion, overlay_occlusion] = graphical_occlusion(net, im, im_,box_small,{16, 8});
save(matName, 'heatmap_deconv', 'mask_deconv', 'heatmap_occlusion', 'mask_occlusion', 'overlay_occlusion');


end
