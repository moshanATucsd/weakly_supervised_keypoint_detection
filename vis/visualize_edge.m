function visualize_edge(points,visibility,edge)

for i=1:size(edge,1)
   p1 = edge(i,1);
   p2 = edge(i,2);
   
   x1 = points(p1,1);
   x2 = points(p2,1);
   y1 = points(p1,2);
   y2 = points(p2,2);
   
   if visibility(p1) && visibility(p2)
       %plot([x1,x2],[y1,y2],'LineStyle','-','color','w','linewidth',7);
       plot([x1,x2],[y1,y2],'LineStyle','-','color','r','linewidth',5);
   else
       
       plot([x1,x2],[y1,y2],'LineStyle','--','color','b','linewidth',3);
   end
end
