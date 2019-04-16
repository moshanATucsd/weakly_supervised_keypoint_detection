function visualize_correspondence(u,v,landmark,u_color,v_color,line_color)

x1 = u(:,1);
y1 = u(:,2);
x2 = v(:,1);
y2 = v(:,2); 

plot([x1,x2]',[y1,y2]','LineStyle','-','color',line_color,'linewidth',3);

plot(u(:,1),u(:,2),'o', ...
'MarkerFaceColor',u_color,'Linewidth',1,'MarkerSize',12,'MarkerEdgeColor','k');

plot(v(:,1),v(:,2),'o', ...
'MarkerFaceColor',v_color,'Linewidth',1,'MarkerSize',12,'MarkerEdgeColor','k');
