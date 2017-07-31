function cMass = computeCenterMass(faces, vertices)

cMass = zeros(size(faces,1),size(vertices,2));
for f = 1:size(faces,1)
   cMass(f,:) = (vertices(faces(f,1),:) + vertices(faces(f,2),:) + vertices(faces(f,3),:) )./3;
end