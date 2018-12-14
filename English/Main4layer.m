clear ; close all; clc

%% Setup the parameters that we will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 300;   % 300 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

X = loadMNISTImages('train-images.idx3-ubyte')';
y = loadMNISTLabels('train-labels.idx1-ubyte');
y = zeroto10(y);
% Xtest = X(50001:end,:);
% ytest = y(50001:end,:);
% 
% X = X(1:50000,:);
% y = y(1:50000,:);

Xtest = loadMNISTImages('t10k-images.idx3-ubyte')';
ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');
ytest = zeroto10(ytest);
fprintf('Loading and Visualizing Data ...\n')
% load('dataset.mat');
% X = (X./255);
% Xval = (Xval./255);
% Xtest = (Xtest./255);
m = size(X, 1);
fprintf('Randomly select 100 data points to display...\n');
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ================ Part 1: Initializing Pameters ================
%  In this part, we will be starting to implment a four
%  layer neural network that classifies digits. We will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ;  initial_Theta3(:)];



%% =================== Part 2: Training NN ===================
%  We have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg".
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 10);

lambda = 0.82;

% Creating cost function object
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1, Theta2 and Theta3 back from nn_params

a = hidden_layer_size * (input_layer_size + 1);

Theta1 = reshape(nn_params(1:a), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
b =  a + (hidden_layer_size * (hidden_layer_size + 1));

Theta2 = reshape(nn_params( (1 + a): b), ...
                 hidden_layer_size, (hidden_layer_size + 1));
Theta3 = reshape(nn_params( (1 + b): end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
% pause;


%% ================= Part 3: Visualize Weights =================
%  We can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:,2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% pred = predict(Theta1, Theta2, Theta3, Xval);
% 
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yval)) * 100);

pred = predict(Theta1, Theta2, Theta3, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

% Manually checking our errors (Error analysis)

check = pred == ytest;
in = find(~check);      %find zero index
pred_wrong = pred(in);
ytest_wrong = ytest(in);
wrong = [pred_wrong , ytest_wrong];   %wrong predictions 

%% ================= Part 5: Saving data  =================
% In the last we will save our paremeters to use it for later purposes
%
 %save("theta.mat",'Theta1','Theta2','Theta3');