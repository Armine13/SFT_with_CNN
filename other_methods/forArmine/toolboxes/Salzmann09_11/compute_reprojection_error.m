function [me,err] = compute_reprojection_error(Cc,BC,VId,UV,A)

err = zeros(size(BC,1),1);
for i=1:size(BC,1)
    P = BC(i,1)*Cc(VId(i,1)+1,:) + BC(i,2)*Cc(VId(i,2)+1,:) + BC(i,3)*Cc(VId(i,3)+1,:);
    uvw = A*P';
    du = (uvw(1)/uvw(3) - UV(i,1));
    dv = (uvw(2)/uvw(3) - UV(i,2));
    err(i) = sqrt(du*du + dv*dv);
end

me = mean(err);