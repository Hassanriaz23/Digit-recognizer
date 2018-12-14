function [pred] = pred_number(im_file)

load('theta.mat');
shiftedImage = center_move(im_file);

shiftedImage = uint8(255) - shiftedImage;     %complement image
shiftedImage = imresize(shiftedImage,[28,28]);
imshow(shiftedImage)
shiftedImage = shiftedImage(:);
shiftedImage = double(shiftedImage)/127.5 - 1;
a1 = shiftedImage;

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