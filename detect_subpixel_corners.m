%subpixel feature coordinate using the Forstner corner point operator
function [corners, corners_sub, scores]=detect_subpixel_corners...
    (nms_img,eig_window_size,Ix,Iy,Ix2,IxIy,Iy2)

[rows, cols] = size(nms_img);
half_win_size = floor(eig_window_size/2);

corners = [];
corners_sub = [];
scores = [];

for i = half_win_size+1:rows-half_win_size
    for j= half_win_size+1:cols-half_win_size
        if nms_img(i,j) ~= 0
            
            N = [Ix2(i, j), IxIy(i, j); IxIy(i, j), Iy2(i, j)];
            
            y_range = i-half_win_size:i+half_win_size;%row-y
            x_range = j-half_win_size:j+half_win_size;%col-x
            Ix2_win = Ix(y_range,x_range).^2;
            Ix2_x_win = Ix2_win.*repmat(x_range,size(Ix2_win,1),1);
            IxIy_win = Ix(y_range,x_range).*Iy(y_range,x_range);
            IxIy_x_win = IxIy_win.*repmat(x_range,size(IxIy_win,1),1);
            IxIy_y_win = IxIy_win.*repmat(y_range,size(IxIy_win,1),1);
            Iy2_win = Iy(y_range,x_range).^2;
            Iy2_y_win = Iy2_win.*repmat(y_range,size(Iy2_win,1),1);
            b = [sum(sum(Ix2_x_win))+sum(sum(IxIy_y_win));...
                sum(sum(IxIy_x_win))+sum(sum(Iy2_y_win))];
            %solve for least squares solution
            corner_sub = N \ b;%[col;row]
            corners_sub = [corners_sub;corner_sub'];
            %integer corner locations
            corner = [j;i];
            corners = [corners;corner'];
            score = nms_img(i,j);
            scores = [scores;score];
        end
    end
end

end