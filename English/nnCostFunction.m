function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1, Theta2 and Theta3, the weight matrices
% for our 4 layer neural network
a = hidden_layer_size * (input_layer_size + 1);

Theta1 = reshape(nn_params(1:a), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
b =  a + (hidden_layer_size * (hidden_layer_size + 1));

Theta2 = reshape(nn_params( (1 + a): b), ...
                 hidden_layer_size, (hidden_layer_size + 1));
Theta3 = reshape(nn_params( (1 + b): end), ...
                 num_labels, (hidden_layer_size + 1));
                 

% Setup some useful variables
m = size(X, 1); % no of cols
          
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
a3 = [ones(m,1) a3];
z4 = a3 * Theta3';
a4 = sigmoid(z4);
theta1_withoutbais = Theta1(:,2:size(Theta1,2));
theta2_withoutbais = Theta2(:,2:size(Theta2,2));
theta3_withoutbais = Theta3(:,2:size(Theta3,2));

for k = 1:num_labels 
  h_k = a4(:,k);
  %y_for_k = find(y == k);
  %x_k = X(k_equal_y,:); 
  y_for_k = (y==k);
  J = J + sum( ((y_for_k) .* log(h_k))+((1-y_for_k) .* log(1-h_k)) );
end
J = -(J/m) + (lambda/(2*m))*(+ sum(sum(theta1_withoutbais .^ 2)) + sum(sum(theta2_withoutbais .^ 2)) + sum(sum(theta3_withoutbais .^ 2)));


delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
delta3 = zeros(size(Theta2));

sigma_4 = zeros(m,num_labels);

for k = 1:num_labels
sigma_4(:,k) = a4(:,k) - (y==k);  
end

sigma_3 = (sigma_4 * Theta3) .* a3 .* (1 - a3);
sigma_3 = sigma_3(:,2:end);

sigma_2 = (sigma_3 * Theta2) .* a2 .* (1 - a2);
sigma_2 = sigma_2(:,2:end);

delta1 = sigma_2' * a1;
delta2 = sigma_3' * a2;
delta3 = sigma_4' * a3;

Theta1_grad(:,1) = (delta1(:,1))/m;
Theta1_grad(:,2:end) = (delta1(:,2:end))/m + (Theta1(:,2:end))*(lambda/m);

Theta2_grad(:,1) = (delta2(:,1))/m;
Theta2_grad(:,2:end) = (delta2(:,2:end))/m + (Theta2(:,2:end))*(lambda/m);

Theta3_grad(:,1) = (delta3(:,1))/m;
Theta3_grad(:,2:end) = (delta3(:,2:end))/m + (Theta3(:,2:end))*(lambda/m);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
