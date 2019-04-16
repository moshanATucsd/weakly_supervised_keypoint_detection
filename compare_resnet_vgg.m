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
car = 'BMW_5/';
imgDir = fullfile(rootDir_data, '/dataset/',car);
images = dir(fullfile(rootDir_data, '/dataset/',car,'*.jpg'));
images = cellfun(@(x) x,{images.name},'UniformOutput',false);

%%

rootDir_net = '/Users/moshan/Documents/Research/cse291_class_project';
net1 = load(fullfile(rootDir_net,'/models/imagenet-vgg-verydeep-19.mat'));
net2 = DDagNN.loadobj(fullfile(rootDir_net,'/models/imagenet-resnet-50-dag.mat'));

%%
resultsDir1 = 'heatmap_vgg/';
resultsDir2 = 'heatmap_resnet/';

% loop over each image
for i=1:5
    
    %%
    % find corresponding index in ground truth file
    index = find(cellfun(@(x) strcmp(x.filename,images(i)),manualParam));
    
    % load image
    filename = manualParam{index}.filename;
    img = imread(fullfile(imgDir,filename));
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
    close;
    box = kp2box(img, kp_visible);
    %%
    overlay1 = occlusion_with_box(net1,img,box);
    overlay2 = occlusion_with_box(net2,img,box);    
    %% VGG
    
    figure;
    imshow(overlay1);
    hold on
    scatter(kp_visible(:,1),kp_visible(:,2),100,'filled','g');
    %% save images
    
    iptsetpref('ImshowBorder','tight');
    set(gca,'LooseInset',get(gca,'TightInset'))
    saveas(gca,fullfile(imgDir,resultsDir1,filename))
    close all;
    
    %% resnet
    
    figure;
    imshow(overlay2);
    hold on
    scatter(kp_visible(:,1),kp_visible(:,2),100,'filled','g');
    %% save images
    
    iptsetpref('ImshowBorder','tight');
    set(gca,'LooseInset',get(gca,'TightInset'))
    saveas(gca,fullfile(imgDir,resultsDir2,filename))
    close all;    
end

function [ overlay ] = occlusion_with_box(net,im,box)
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

%{8, 8} or {16, 16}
[heatmap_occlusion, mask_occlusion, overlay] = graphical_occlusion(net, im, im_,box_small,{16, 16});

end

