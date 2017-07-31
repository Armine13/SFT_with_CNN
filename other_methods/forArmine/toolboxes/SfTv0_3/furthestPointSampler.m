function rInds = furthestPointSampler(X,numpts,ptsPrev)
numData = size(X,2);
rInds= zeros(numpts,1);
if isempty(ptsPrev)
rr = randperm(numData); rr=rr(1);
rInds(1) = rr;
X_ = X(:,rr);
ds = distance_vec(X_,X);

else
    dd = distance_vec(ptsPrev,X);
    ds = min(dd,[],1);  
    rInds = [];
end
%ds(rr) = inf;

for i=1:numpts
    [mm,mn] = max(ds);
    rInds(i+1) = mn;
    %X_ = [X_,X(:,mn)];
    ds_ = distance_vec(X(:,mn),X);
    ds = min([ds;ds_],[],1);  
end
