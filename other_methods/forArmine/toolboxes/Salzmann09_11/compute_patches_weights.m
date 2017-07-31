function [W] = compute_patches_weights(Nr,noUse,nM,w)

Nr(noUse,:) = 0;
ws = sum(Nr);

%mw = min(ws(find(ws)));

%ws(find(ws==0)) = 1;

med = median(ws(find(ws)));
ws = exp(-ws./med).*w;

% ws = ws - min(ws);
% mw = max(ws);
% if(mw > 0)
%     ws = ws ./ max(ws);
%     ws = (1-ws).*w;
% end

ws = reshape(repmat(ws,nM,1),nM*length(ws),1); 
%ws = reshape(repmat(ws,3*25+nM,1),(3*25+nM)*length(ws),1); 
W = sparse(diag(ws));

%fprintf('Min wr: %d, Max wr: %d\n', min(ws), max(ws));
