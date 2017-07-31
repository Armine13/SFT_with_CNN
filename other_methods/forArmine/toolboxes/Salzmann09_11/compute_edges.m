function E = compute_edges2(tri,nPts)

Adj = zeros(nPts,nPts);
for i=1:size(tri,1)
    Adj(tri(i,1)+1,tri(i,2)+1) = 1;
    Adj(tri(i,1)+1,tri(i,3)+1) = 1;
    Adj(tri(i,2)+1,tri(i,1)+1) = 1;
    Adj(tri(i,2)+1,tri(i,3)+1) = 1;
    Adj(tri(i,3)+1,tri(i,1)+1) = 1;
    Adj(tri(i,3)+1,tri(i,2)+1) = 1;
end

E = zeros(0,2);
ind = 1;
for i=1:nPts
    for j=i+1:nPts
        if(Adj(i,j)==1)
            E(ind,:) = [i,j];
            ind = ind+1;
        end
    end
end

