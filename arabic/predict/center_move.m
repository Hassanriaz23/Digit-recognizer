function  shiftedImage = center_move(im_file)

im = imread(im_file);
im = rgb2gray(im);
im_binary = imcomplement(im);
im_binary = im_binary>0;

%%
measurements = regionprops(im_binary, 'Centroid');
[rows, columns] = size(im_binary);
rowsToShift = round(rows/2- measurements.Centroid(2));
columnsToShift = round(columns/2 - measurements.Centroid(1));

% Call circshift to move region to the center.
shiftedImage = circshift(im, [rowsToShift columnsToShift]);

imshow(shiftedImage)
end