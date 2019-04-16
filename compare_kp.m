function [detected_kp, kp_label, detected_gt_kp] = compare_kp(box, keypoints, kp_visible, landmark_visible)

detected_kp = [];
kp_label = [];
detected_gt_kp = [];

alpha = 0.2;%0.1 or 0.2 in paper
radius = alpha*(max(box(3), box(4)));

num_kp = size(keypoints,1);
num_landmark = size(kp_visible,1);

for i = 1:num_kp
    kp = keypoints(i,:);
    kps = repmat(kp,num_landmark,1);
    distances = diag((kps-kp_visible)*(kps-kp_visible)');
    [min_dist, min_ind] = min(distances);
    if min_dist < radius
        detected_kp = [detected_kp;kp];
        kp_label = [kp_label;landmark_visible(min_ind)];
        detected_gt_kp = [detected_gt_kp;kp_visible(min_ind,:)];
    end
end

end

