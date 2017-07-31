function Q=b2p(vertexPos,faces,b)

P1=vertexPos(faces(:,1),:);
P2=vertexPos(faces(:,2),:);
P3=vertexPos(faces(:,3),:);

Q=P1'.*([1;1;1]*b(:,1)')+P2'.*([1;1;1]*b(:,2)')+P3'.*([1;1;1]*b(:,3)');


