% Export Camera parameters
load model
load Calib_Results kc KK;
   
K=KK;
T=model.camSeq.CameraDevices{1}.T;
R=model.camSeq.CameraDevices{1}.R;
%kc=[model.camSeq.CameraDevices{1}.k1,model.camSeq.CameraDevices{1}.k2,model.camSeq.CameraDevices{1}.p1,model.camSeq.CameraDevices{1}.p2,model.camSeq.CameraDevices{1}.k3];
vertexPos=model.surfaceMesh.vertexPos;
faces=model.surfaceMesh.faces;
savevar(K,'views_undistorted/K.cmat');
savevar(R,'views_undistorted/R.cmat');
savevar(T,'views_undistorted/T.cmat');
savevar(kc,'views_undistorted/kc.cmat');
savevar(vertexPos,'views_undistorted/vertexPos.cmat');
savevar(faces,'views_undistorted/faces.cmat');
% Load baricentrics of one frame

for i=[1:length(model.rawFrameNames)]
    stringname=strtok(model.rawFrameNames(i).name,'.tif');    
    T=model.camSeq.CameraDevices{i}.T;
    R=model.camSeq.CameraDevices{i}.R;
    if(length(T)>0)
    eval(sprintf('load views_undistorted/%s_BaryMap.mat',stringname));
    B1=B(:,:,1);
    B2=B(:,:,2);
    B3=B(:,:,3);
    B4=B(:,:,4);
    savevar(B1,sprintf('views_undistorted/%sB1.cmat',stringname));
    savevar(B2,sprintf('views_undistorted/%sB2.cmat',stringname));
    savevar(B3,sprintf('views_undistorted/%sB3.cmat',stringname));
    savevar(B4,sprintf('views_undistorted/%sB4.cmat',stringname));
    savevar(R,sprintf('views_undistorted/%sR.cmat',stringname)); 
    savevar(T,sprintf('views_undistorted/%sT.cmat',stringname));  
    T
    end
end
