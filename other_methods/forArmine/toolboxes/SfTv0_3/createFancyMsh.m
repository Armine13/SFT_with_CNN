function Msh=createFancyMsh(Q,p,img)
Msh.vertexPos=Q';
tri=delaunay(p(1,:),p(2,:));
Msh.faces=tri;
Msh.noVertexPos=size(Q,2);
Msh.noFaces=size(tri,1);
Msh.texMap.vertexUVW=[p(1,:)./size(img,2);1-p(2,:)./size(img,1);zeros(1,size(p,2))]';
Msh.texMap.facesT=tri;
Msh.texMap.matIDs=ones(size(tri,1),1);
Msh.texMap.img=img;
end
