function displayMatches2ImgswithColor(h,psImg1,psImg2,img1,img2)
% This function displays with colors matches between two images
% h: the handle of the figure
if size(psImg1,1) ~= size(psImg2,1) && size(psImg1,2) ~= size(psImg2,2)
    error('The two sets of matches are not of the same size!');
end

cmap = hsv(length(psImg1(2,:)));
figure(h);
clf;
subplot(121);
imshow(img1);
hold on
for j = 1:size(psImg1,2)
    plot(psImg1(1,j),psImg1(2,j),'+','Color',cmap(j,:));
end
hold off;
title('Matches in the first image');
subplot(122);
imshow(img2);
hold on
for j = 1:size(psImg2,2)
    plot(psImg2(1,j),psImg2(2,j),'+','Color',cmap(j,:));
end
hold off;
title('Matches in the second image');

end