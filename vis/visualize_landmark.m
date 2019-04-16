function [kp_visible, landmark_visible] = visualize_landmark(points,visibility,landmark,color)
%landmark_visible stores the index
%kp_visible stores the point locations

plot_flag = 1;

temp = zeros(length(points),1);
temp(landmark) = 1;

if isempty(visibility)
    points_tmp = points(temp==1,:);
    if (plot_flag == 1)
        plot(points_tmp(:,1),points_tmp(:,2),'o','MarkerFaceColor', ...
            color,'Linewidth',1,'MarkerSize',12,'MarkerEdgeColor','k');
    end
else
    points_tmp = points(temp==1 & visibility==1,:);
    if (plot_flag == 1)
        plot(points_tmp(:,1),points_tmp(:,2),'o','MarkerFaceColor', ...
            color,'Linewidth',1,'MarkerSize',12,'MarkerEdgeColor','k');
    end
    kp_visible = points_tmp;
    landmark_visible = find(temp.*visibility);
    
    if (plot_flag == 1)
        points_tmp = points(temp==1 & visibility==0,:);
        plot(points_tmp(:,1),points_tmp(:,2),'o','Linewidth',1,'MarkerSize',12,'MarkerEdgeColor','w');
    end
end

if (plot_flag == 1)
    text(points(landmark,1),points(landmark,2), ...
        num2str(landmark),'BackgroundColor',color,'FontSize',12);
end