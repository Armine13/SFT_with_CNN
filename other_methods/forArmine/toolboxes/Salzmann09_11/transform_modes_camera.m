function PC = transform_modes_camera(PC,Rt)

n = size(PC,1)/3;

for i=1:size(PC,2)
    tmp = reshape(PC(:,i),3,n);
    tmp = Rt(:,1:3)*tmp;
    PC(:,i) = reshape(tmp,3*n,1);
end
