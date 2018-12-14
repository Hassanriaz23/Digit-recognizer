clc;
clear;
% BackgroundImage = imread('back.png');
object = imread('931.png');
% BackgroundImage = ones(size(object)) * (255);


subplot(3,3,1);
imshow(object);
title('Orignal Image');
ga = rgb2gray(object);
BW = im2bw(ga);
subplot(3,3,2);
imshow(BW)
title('convert im2bw');

% gb = rgb2gray(BackgroundImage);
gb = uint8(ones(size(ga))) * (255);
foregroundDetector = vision.ForegroundDetector('InitialVariance',(30/255)^2);
foreground = step(foregroundDetector, gb);
foreground1 = step(foregroundDetector, ga);

BlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',100,'MaximumBlobArea',50000);
[area,centroid,bbox] = step(BlobAnalysis,foreground1);
Ishape = insertShape(object,'rectangle',bbox,'Color', 'green','Linewidth',6);

subplot(3,3,3);
imshow(Ishape);


no_of_digits = size(bbox,1);
pred = zeros(1,no_of_digits);

for k = 1:no_of_digits
    im_k = imcrop(ga,bbox(k,:));
    % resize based on scale:
    maxSize = max(size(im_k));
    im_k = imresize(im_k,20/maxSize);
    
    white = 255 * ones(28, 28, 'uint8');
    % assign the image inside the white one:
    % get the size of the resized image
    sizeK = size(im_k);
    % determine the indices to place it in the middle of the padded image
    iStart = floor((28-sizeK(1))/2)+1;
    iEnd = iStart+sizeK(1)-1;
    jStart = floor((28-sizeK(2))/2)+1;
    jEnd = jStart+sizeK(2)-1;
    white(iStart:iEnd,jStart:jEnd) = im_k;
    
    final_image = white;
    subplot(3,3,k+3)
    imshow(final_image)
    pred(k) = pred_number1(final_image);
    imshow(final_image)
end
fullnumber = polyval(pred, 10)