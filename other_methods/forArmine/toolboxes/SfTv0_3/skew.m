function s = skew( u )

s = zeros(3,3);

s(1,2) = -u(3);   s(1,3) =  u(2);  
s(2,1) =  u(3);   s(2,3) = -u(1);
s(3,1) = -u(2);   s(3,2) =  u(1);
