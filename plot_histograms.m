clear;clc;close all;

addpath('vis');
addpath('fitting');
addpath('common');

addpath('tSNE');

%%

rootDir_data = '/Users/moshan/Documents/Research/cse291_class_project/datasets/FG3DCar';

% load face
load(fullfile(rootDir_data,'/3dmodel/car_face_mesh.mat'));
face = carFace_mesh;
% load landmark
landmark = textread(fullfile(rootDir_data,'/3dmodel/landmark_ext.txt'),'%f');
num_landmark = numel(landmark);
% load ground truth
load(fullfile(rootDir_data,'/3dmodel/ground_truth.mat'),'manualParam');

% get image list
car = 'original/';
imgDir = fullfile(rootDir_data, '/dataset/',car);
images = dir(fullfile(rootDir_data, '/dataset/',car,'*.jpg'));
images = cellfun(@(x) x,{images.name},'UniformOutput',false);

%%

histDir = 'histograms/';
histDir = fullfile(rootDir_data, '/dataset/',histDir);

%%
num_images = length(images);
% num_images = 150;
landmark_histogram_all = zeros(num_images, num_landmark);
labels_all = cell(num_images,1);
% loop over each image
for i=1:num_images
    %%
    % find corresponding index in ground truth file
    index = find(cellfun(@(x) strcmp(x.filename,images(i)),manualParam));
    
    % load image
    filename = manualParam{index}.filename;
    histName = fullfile(histDir,strcat(filename(1:end-4),'_hist.mat'));
    fprintf('Processing %s\n', filename);    
    
    load(histName);
    landmark_histogram_all(i,:) = landmark_histogram;
    
    k = strfind(filename,'_');
    labels_all{i} = filename(1:k(1)-1);
end

histograms_each_car = zeros(30,64);
for j=1:30
    histograms_each_car(j,:) = sum(landmark_histogram_all((j-1)*10+1:j*10,:),1);
end
[feature_num_each, feature_ind_each] = max(histograms_each_car,[],2);
landmark_ind_each = landmark(feature_ind_each);

X = sum(landmark_histogram_all,1);
[feature_num_all, feature_ind_all] = max(X);
landmark_ind_all = landmark(feature_ind_all);