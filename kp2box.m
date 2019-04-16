function box = kp2box(img, kp_visible)
%

[h,w] = size(img);

top_left = [min(kp_visible(:,1)), min(kp_visible(:,2))];
width = max(kp_visible(:,1)) - min(kp_visible(:,1));
height = max(kp_visible(:,2)) - min(kp_visible(:,2));

% margin = 5;%too tight
margin = 15;

top_left(1) = max(top_left(1)-margin,0);
top_left(2) = max(top_left(2)-margin,0);

if top_left(1)+width+2*margin < w
    width = width+2*margin;
end

if top_left(2)+height+2*margin < h
    height = height+2*margin;
end

box = [top_left(1), top_left(2), width, height];

end

