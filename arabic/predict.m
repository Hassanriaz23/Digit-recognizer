function p = predict(Theta1, Theta2, Theta3, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta3, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
[~, p] = max(h3, [], 2);

% =========================================================================


end
