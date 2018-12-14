function [pred] = pred_number1(im)
load('arabic_theta.mat');

im = uint8(255) - im;     %complement image
imshow(im)
im = im(:);
im = double(im)/127.5 - 1;
a1 = im;

  a1 = [ 1; a1];
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1;a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  a3 = [1;a3];
  z4 = Theta3 * a3;
  a4 = sigmoid(z4);
  
[~, pred] = max((a4));
if pred == 10
    pred = 0;
end
end