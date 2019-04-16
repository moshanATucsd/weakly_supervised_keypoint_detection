% set default parameters
function param = create_configuration()

%% general
param.class_num             = 30;
param.image_num             = 300;
param.img_dir               = 'dataset/NetCarShow300/original';
param.fg_resize_dir         = 'dataset/NetCarShow300/resize';        
param.image_height          = 300;
param.patch_size            = 55;

%% hog descriptors
param.cell_size             = 8;
param.orientation_num       = 9;
param.hog_w                 = floor((param.patch_size + param.cell_size/2) / param.cell_size);
param.hog_h                 = floor((param.patch_size + param.cell_size/2) / param.cell_size);
param.hog_size              = param.hog_w * param.hog_h * param.orientation_num * 4;
