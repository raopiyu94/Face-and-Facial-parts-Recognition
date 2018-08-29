%% Simple Face Recognition Example
%% Load Image Information from ATT Face Database Directory
faceDatabase = imageSet('FaceDatabaseATT','recursive');

%% Display All Images of First Face
figure;
montage(faceDatabase(1).ImageLocation); % used to club all images
title('Images of Single Face');

%%  Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
for i=1:size(faceDatabase,2)
imageList(i) = faceDatabase(i).ImageLocation(4); % displaying 4th image out of the 10 given images of 40 samples
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
diff = zeros(1,9);
title('Query Image v/s Complete Database');

%% Facial Parts Detection
personToQuery = 13;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
a = galleryImage;
detector = vision.CascadeObjectDetector; 
% function used to detect various parts of upper facial body
detector1 = vision.CascadeObjectDetector('mouth');
detector1.MergeThreshold = 100; 
% since mouth is unclear and is getting detected at 1 more wrong place
detector2 = vision.CascadeObjectDetector('EyePairBig');
detector3 = vision.CascadeObjectDetector('Nose');
bbox = step(detector,a);
bbox1 = step(detector1,a);
bbox2 = step(detector2,a);
bbox3 = step(detector3,a);
out = insertObjectAnnotation(a,'rectangle',bbox,'detection');
out1 = insertObjectAnnotation(a,'rectangle',bbox1,'mouth');
out2 = insertObjectAnnotation(a,'rectangle',bbox2,'eyes');
out3 = insertObjectAnnotation(a,'rectangle',bbox3,'nose');
subplot(1,4,1);imshow(out);
subplot(1,4,2);imshow(out1);
subplot(1,4,3);imshow(out2);
subplot(1,4,4);imshow(out3);

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2]);


%% Extract and display Histogram of Oriented Gradient Features for single face 
person = 34;
[hogFeature, visualization] = extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% Extract HOG Features for training set 
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc 
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Match First 5 People from Test Set
figure;
figureNum = 1;
for person = 1:5
    for j = 1:test(person).Count
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');
        figureNum = figureNum+2;
        
    end
    figure;
    figureNum = 1;

end



