function C = getVToXMatrix(bcs,vids,nPts)

nbrpoints = size(bcs,1);

C = sparse(zeros(3*nbrpoints,3*nPts));
for i=1:nbrpoints
    C(3*i-2:3*i,3*vids(i,1)+1:3*vids(i,1)+3) = bcs(i,1)*speye(3);
    C(3*i-2:3*i,3*vids(i,2)+1:3*vids(i,2)+3) = bcs(i,2)*speye(3);
    C(3*i-2:3*i,3*vids(i,3)+1:3*vids(i,3)+3) = bcs(i,3)*speye(3);
end
