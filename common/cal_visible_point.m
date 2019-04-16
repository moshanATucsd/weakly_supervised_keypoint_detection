function visible_point = cal_visible_point(visible_faces,faces,point_num)

visible_point = zeros(point_num,1);

for i=1:length(faces)
    if ~isempty(find(visible_faces == i,1))    
        face = faces{i};
        visible_point(face) = 1;
    end
end