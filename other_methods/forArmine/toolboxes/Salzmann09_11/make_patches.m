function [patches,wvs] = make_patches(sx,sy,p,q,n,m,step)

ind = 1;
for j=sy:step:m-q
    for i=sx:step:n-p
        for k=1:q
            patches(ind,(k-1)*p+1:k*p) = [i:i+p-1]+(j+k-1)*n;
        end
        ind = ind+1;
    end
end

wvs = zeros(1,n*m);
for i=1:size(patches,1)
    for j=1:size(patches,2)
        wvs(patches(i,j)+1) = wvs(patches(i,j)+1)+1;
    end
end
wvs = 1./wvs;
