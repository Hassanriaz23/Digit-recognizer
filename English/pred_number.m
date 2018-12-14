function pred = pred_number(image,Theta1,Theta2)
  
imfile = image;;
%im = rgb2gray(imread(imfile)); % double and convert to grayscale
im = imread(imfile);
im = imcomplement(im);
im = imresize(im,[28,28]);
im = im(:);
im = double(im);
im = im./127.5 - 1;
im = [ 1; im];
  z2 = Theta1 * im;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  [cc, pred] = max((a3));
  
  
  
  
endfunction