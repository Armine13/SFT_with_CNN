function [Nr] = compute_patches_matches(matches,patches)

nP = size(patches,1);
nM = size(matches,1);
Nr = zeros(nM,nP);

for i=1:nP
    for j=1:nM
        inds = intersect(matches(j,1:3),patches(i,:));
        if(length(inds) == 3)
            Nr(j,i) = 1;
        end
    end
end
