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
    labels_all{i} = filename(1:k(2)-1);
end

X = sum(landmark_histogram_all,1);
bar(X);

% % Set parameters
% no_dims = 2;
% initial_dims = 10;
% perplexity = 30;
% 
% % Run t?SNE
% mappedX = tsne(landmark_histogram_all, [], no_dims, initial_dims, perplexity);
% 
% % Plot results
% gscatter(mappedX(:,1), mappedX(:,2), labels_all,[],'ox+*sdv^<>ph.');

% colors = 'rgb';
% markers = 'osd';
% 
% for idx = 1 : 3
%     plot3(mappedX(:,1), mappedX(:,2), mappedX(:,3), [colors(idx) markers(idx)]);
%     hold on;
% end
% grid; %// Show a grid