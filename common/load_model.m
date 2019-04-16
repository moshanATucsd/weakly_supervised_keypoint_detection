function [face,edge,landmark] = load_model()

% load face
load('../3dmodel/car_face_mesh.mat');
face = carFace_mesh;

% load edge
[edge] = textread('../3dmodel/salient_edge.txt');
edge = [edge;edge+128];

% load landmark
landmark = textread('../3dmodel/landmark_ext.txt','%f');