function [A,c,q] = socpConstraints(edges,nPts)

nedges = size(edges,1);
q = 4*ones(1,nedges);
A=sparse(4*nedges,3*nPts);
c=sparse(4*nedges,1);
    
for jj=1:nedges
    c(4*jj-3) = edges(jj,3);
    A(4*jj-2:4*jj,3*edges(jj,2)-2:3*edges(jj,2)) = -speye(3);
    A(4*jj-2:4*jj,3*edges(jj,1)-2:3*edges(jj,1)) = speye(3);
end
