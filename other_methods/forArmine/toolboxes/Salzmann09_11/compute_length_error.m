function [lerr,merr] = compute_length_error(Cc, E, L)

errs = zeros(size(E,1),1);

for i=1:size(E,1)
    i1 = E(i,1);
    i2 = E(i,2);
    
    t1 = Cc(i2,:) - Cc(i1,:);
    errs(i) = abs(sqrt(t1*t1') - L(i));
end

lerr = mean(errs);
merr = max(errs);