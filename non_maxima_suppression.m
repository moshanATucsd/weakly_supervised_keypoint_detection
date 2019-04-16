%non maxima suppression
function [nms_img] = non_maxima_suppression( image, window_size)

[rows, cols] = size(image);
nms_img = zeros(size(image));

half_win_size = floor(window_size/2);
stride = 1;

for i = half_win_size+1:stride:rows-half_win_size
    for j= half_win_size+1:stride:cols-half_win_size
        if image(i, j)==max(max(image(i-half_win_size:i+half_win_size,...
                j-half_win_size:j+half_win_size)))
            nms_img(i, j) = image(i, j);
            image(i, j) = image(i, j)+0.1;
        end
    end
end

%sanity check
% figure;
% nms_show = zeros(size(nms_img));
% nms_show(nms_img~=0) = 1;
% imshow(mat2gray(nms_show));

end