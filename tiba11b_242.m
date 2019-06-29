clear all
close all
%% video reader
t=0;

videoFileReader = vision.VideoFileReader('video.wmv');
videoFrame = step(videoFileReader);

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',[300 300 videoInfo.VideoSize+30]);

while ~isDone(videoFileReader)
tic
    t=t+1;
    % Extract the next video frame
    videoFrame = step(videoFileReader);

    % RGB -> Gray
    i=rgb2gray(videoFrame);
%% clear
clear o
%% face detect
faceDetector = vision.CascadeObjectDetector;
bbox = step(faceDetector, i);
if numel(bbox)==0
    continue
end
%% low pass filter
sig=0.005*bbox(3);
h=fspecial('gaussian',[5 5],sig);
I=imfilter(i,h);
%% eye crop
rightEye=imcrop(I,[0.2*bbox(3)+bbox(1) ...
    0.4*bbox(4)-0.03*bbox(3)+bbox(2) 0.22*bbox(3) 0.15*bbox(3)]);
%% Gradient
[Gxr, Gyr] = imgradientxy(rightEye);
[Gmagr, ~] = imgradient(Gxr, Gyr);

rightTresh=0.3*std2(Gmagr) + mean2(Gmagr);

Gxr(Gmagr < rightTresh)=0;
Gyr(Gmagr < rightTresh)=0;

Gmagr(Gmagr < rightTresh)=0;
% Gdirr(Gmagr < rightTresh)=0;

Gxr_=Gxr./Gmagr;
Gyr_=Gyr./Gmagr;

Gxr_(isnan(Gxr_))=0;
Gyr_(isnan(Gyr_))=0;
%% 

[hr,wr]=size(rightEye);
Wcr=im2double(255-rightEye);
pixs=hr*wr;
for w=1:wr,for h=1:hr,o(h,w)=h+hr*(w-1);end,end
[y,x,~]=ind2sub([hr,wr],o);
dotpro=zeros(hr,wr);

for c=1:pixs
    [cy,cx]=ind2sub([hr,wr],c);
    dx=x - cx;
    dy=y - cy;
    dmag=sqrt((dx.*dx) + (dy.*dy));
    dx_=dx./dmag;
    dy_=dy./dmag;
    dx_(isnan(dx_))=0;
    dy_(isnan(dy_))=0;
    dotpro_=(dx_.*Gxr_)+(dy_.*Gyr_);
    dotpro_(dotpro_ < 0)=0;
    dotpro_=(dotpro_ .* dotpro_).*Wcr;
    dotpro=dotpro + dotpro_;
end
%% argmax
colr_=sort(dotpro);
[~,colr_]=max(colr_(end,:));
rowr_=sort(dotpro');
[~,rowr_]=max(rowr_(end,:));
%% rescale
rowr=rowr_ + 0.4*bbox(4)-0.03*bbox(3)+bbox(2);
colr=colr_ + 0.2*bbox(3)+bbox(1);

videoOut1 = insertObjectAnnotation(videoFrame,'circle',[colr,rowr,4],' ');
step(videoPlayer,double(videoOut1));
imshow(rightEye),hold on,plot(colr_,rowr_,'rx','MarkerSize',5),hold off
z(t)=toc;
end

time=sum(z)
frameRate=t/time