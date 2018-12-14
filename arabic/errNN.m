function [J] = errNN(Theta1, Theta2, Theta3, X, y)
  
  m = length(y);
  h = predict(Theta1, Theta2, Theta3, X);
  err = (h!=y);
  J = (1/m) * sum(err); 
  
endfunction