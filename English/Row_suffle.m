function [X1,y1] = Row_suffle(X,y);
  [m n] = size(X);
   X1 = zeros(m,n);
   y1 = zeros(m,1);
   idx = randperm(m);
   y1 = y(idx);  
   X1 = X(idx,:);
end
