function [x_center,y_center,ind] = pupil_center2(eye_img)
if size(size(eye_img),2)>2
    I = im2uint8(histeq(rgb2gray(eye_img)));
else
    I = im2uint8(histeq(eye_img));
end
% cd('D:\Documents\MATLAB\matlab\matlab')
cd('C:\users\aryan_461\Documents\MATLAB\matlab\matlab')
modfile = 'C:\users\aryan_461\Documents\MATLAB\matlab\matlab\shape_predictor_68_face_landmarks.dat';
find_face_landmarks(modfile);
% if max(size(I))>720
%     I = imresize(I,0.5);
% end
f = find_face_landmarks(modfile,I);
if ~isempty(f.faces)
    ind = f.faces.landmarks(37:42,:);
    img = I(min(ind(:,2)):max(ind(:,2)) , min(ind(:,1)):max(ind(:,1)),:);
else
    x_center = 0;
    y_center = 0;
    ind = [];
    return
end

[Gmag,~] = imgradient(img);
[Gx, Gy] = imgradientxy(img);
Gmag(Gmag < max(max(Gmag))*0.1)=0;
Gx = Gx./Gmag;
Gy = Gy./Gmag;
Gx(isnan(Gx))=0;
Gy(isnan(Gy))=0;
Gx(isinf(Gx))=0;
Gy(isinf(Gy))=0;
x = 1:size(img,2);
y = 1:size(img,1);
[x,y] = meshgrid(x,y);
cx = x; cy = y;
w = (1-im2double(img));
for j=1:numel(img)
    mag = sqrt((x-cx(j)).^2 +(y-cy(j)).^2);
    dix = (x-cx(j))./mag;
    diy = (y-cy(j))./mag;
    dix(isnan(dix))=0;
    diy(isnan(diy))=0;
    c(j) = w(j)*sum(sum((dix.*Gx + diy.*Gy).^2));
end



[~,indmax]=max(c);
% img= insertMarker(img,[cx(indmax),cy(indmax)],'x','color','r');
% imshow(img)

x_center = cx(indmax)+ double(min(ind(:,1)))-1; 
y_center = cy(indmax)+ double(min(ind(:,2)))-1;


end