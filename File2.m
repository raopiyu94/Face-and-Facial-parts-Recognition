%% Simple Face Recognition Example
%% Load Image Information from ATT Face Database Directory
faceDatabase = imageSet('FaceDatabaseATT','recursive');

%% Facial Parts Detection
personToQuery = 13;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
a = galleryImage;
% since mouth is unclear and is getting detected at 1 more wrong place
detector2 = vision.CascadeObjectDetector('EyePairBig');
bbox2 = step(detector2,a);
out2 = insertObjectAnnotation(a,'rectangle',bbox2,'eyes');
xyz = imcrop(a,bbox2);
abc = imresize(xyz,100);
imshow(xyz);
% Gathering the data 
for i=1:5
figure;
imageList = read(faceDatabase(i),1);
% displaying 4th image out of the 10 given images of 40 samples
detectorp = vision.CascadeObjectDetector('EyePairSmall');
bboxp = step(detectorp,imageList);
pqr = imcrop(imageList,bboxp);
imshow(pqr);
if i == 1
    imwrite(pqr,'abc/pqr1.pgm','pgm');
elseif i == 2
    imwrite(pqr,'abc/pqr2.pgm','pgm');
elseif i == 3
    imwrite(pqr,'abc/pqr3.pgm','pgm');
elseif i == 4
    imwrite(pqr,'abc/pqr4.pgm','pgm');
else
    imwrite(pqr,'abc/pqr5.pgm','pgm');
end
end


%% Load Image Information from abc Database Directory
eyesDatabase = imageSet('abc','recursive');
imgg = read(eyesDatabase(1),2);
imshow(imgg);

%% Extract and display Histogram of Oriented Gradient Features for single face 
imgg = read(eyesDatabase(1),2);
[hogFeature, visualization] = extractHOGFeatures(read(eyesDatabase(1),2));
figure;
subplot(2,1,1);imshow(read(faceDatabase(1),4));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

    