function [A,c,q] = socpLocalPrior(P,Vr,nPts)

nP = size(P,1);
nV = size(Vr,1);
if(nP ~= nV) && ~isempty(P)
    q = [(nP+1),(nV+1)];
    A=[sparse([zeros(1,3*nPts),-1]);-1.*[sparse(P(:,1:end-1)),sparse(nP,1)];sparse([zeros(1,3*nPts),-1]);-[sparse(Vr(:,1:end-1)),sparse(nV,1)]];
    c=[sparse(1,1);1.*sparse(P(:,end));sparse(1,1);sparse(Vr(:,end))];
else
    q = (nV+1);
    A=[sparse([zeros(1,3*nPts),-1]);-[sparse(Vr(:,1:end-1)),sparse(nV,1)]];
    c=[sparse(1,1);sparse(Vr(:,end))];
end
