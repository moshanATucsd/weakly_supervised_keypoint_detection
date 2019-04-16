function normal = compute_normal(car_face,car_model)
    if(size(car_model,2) == 1)
        car_model = reshape(car_model,3,[]);
    end
    normal = zeros(3,size(car_model,2));
    
    % left side
    for p = 1:171
        current_face= car_face{p};
        point_num = size(current_face,2);
        for q = 1:point_num
            face(:,q) = car_model(:,current_face(q));
        end
        a = face(:,2) - face(:,1);
        b = face(:,3) - face(:,1);
        c = cross(a,b);
        c = c/norm(c);
        normal(:,p) = c;
    end
    
    % another side
    for p = 172:342
        current_face= car_face{p};
        point_num = size(current_face,2);
        for q = 1:point_num
            face(:,q) = car_model(:,current_face(q));
        end
        a = face(:,2) - face(:,1);
        b = face(:,3) - face(:,1);
        c = cross(b,a);
        c = c/norm(c);
        normal(:,p) = c;
    end
end