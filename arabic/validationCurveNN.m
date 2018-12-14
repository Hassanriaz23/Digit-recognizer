function [lambda_vec, error_train, error_val] = ...
    validationCurveNN(X, y, Xval, yval, initial_nn_params, num_labels)


% Selected values of lambda 
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
  
  lambda = lambda_vec(i);
  [Theta1 , Theta2] = trainNN(X, y, initial_nn_params, lambda);
  
  Jtrain = errNN(Theta1 , Theta2, X, y); 
	Jval = errNN(Theta1 , Theta2, Xval, yval);

   %storing the result 
   error_train(i) = Jtrain;  
   error_val(i) = Jval;
end

% =========================================================================

end