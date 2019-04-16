function visualize_wireframe(points,visible_face,faces)

for i=1:length(faces);
    if ~isempty(find(visible_face == i,1))   
        face = faces{i};
        visible_points = points(face,:);
        visible_points(end+1,:) = visible_points(1,:);  %#ok<AGROW>
        plot(visible_points (:,1),visible_points(:,2), ...
        'LineStyle','-','color', [0.3,0.3,0.3],'linewidth',1);
    end
end