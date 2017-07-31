function M = compute_correspondence_matrix(A, bcs, vid, uvs, step, nPts)

nTot = size(uvs,1);
nMatches = ceil(nTot/step);
M = sparse(2*nMatches,3*nPts);

j = 0;
for i=1:step:nTot
    
    %% Assumes coordinates are ordered [x1,y1,z1,...,xN,yN,zN]
    M(2*j+1:2*j+2,3*vid(i,1)+1:3*vid(i,1)+3) = sparse(bcs(i,1)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]));
    M(2*j+1:2*j+2,3*vid(i,2)+1:3*vid(i,2)+3) = sparse(bcs(i,2)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]));
    M(2*j+1:2*j+2,3*vid(i,3)+1:3*vid(i,3)+3) = sparse(bcs(i,3)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]));
    
    %% Assumes coordinates are ordered [x1,...,xN,y1,...,yN,z1,...,zN]
%     M(2*j+1:2*j+2,[vid(i,1)+1,nPts+vid(i,1)+1,2*nPts+vid(i,1)+1]) = bcs(i,1)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]);
%     M(2*j+1:2*j+2,[vid(i,2)+1,nPts+vid(i,2)+1,2*nPts+vid(i,2)+1]) = bcs(i,2)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]);
%     M(2*j+1:2*j+2,[vid(i,3)+1,nPts+vid(i,3)+1,2*nPts+vid(i,3)+1]) = bcs(i,3)*(A(1:2,:)-[uvs(i,1).*A(3,:);uvs(i,2).*A(3,:)]);
    
    j = j+1;
end
