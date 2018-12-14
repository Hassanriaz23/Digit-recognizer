clear ; close all; clc

%% Setup the parameters that will be used later
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 300;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
% loading data set 
load('data.mat')
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
%  In this part of the code, we will be starting to implment a three
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
%  network. To train your neural network, we will now use "fmincg"

fprintf('\nTraining Neural Network... \n')

% How much iteration fmincg does to minimize cost function
options = optimset('MaxIter', 2000);

%regulization paremeter
lambda = 0.82;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
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

% fprintf('\nVisualizing Neural Network... \n')
% 
% displayData(Theta1(:, 2:end));
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% pred = predict(Theta1, Theta2, Theta3, Xval);
% 
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yval)) * 100);

pred = predict(Theta1, Theta2, Theta3, Xtest);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

%% ================= Part 4: Error analysis =================
%  After prdicting our training test, we would like to manualy check that
%  our algorithm is working fine. We can check some pattern of error also
%  and can modify our code accordinly. In this we see wrong predicted
%  values and compare them by real values

check = (pred == ytest);

%find zero index which we classified wrong
in = find(~check);

pred_wrong = pred(in);
ytest_wrong = ytest(in);

wrong = [pred_wrong , ytest_wrong];

%% ================= Part 5: Saving data  =================
%  After all our errors are reduced, we can save our Theta1, Theta2 and
%  Theta 3 which will be used to predict numbers later.

%  save('arabic_theta.mat','Theta1','Theta2','Theta3');