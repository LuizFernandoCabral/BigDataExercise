%% Initialization
clear ; close all; clc

%% Setup the parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % ("0" to label 10)
lambda = 3; % Can set different values for lambda

%% ---------
% Load Training Data
%% ---------
fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ---------
% NN Parameters
%% ---------
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ---------
% Traininig NN
%% ---------
fprintf('\nTraining Neural Network... \n')

%  Create options structure for optimization functions.
options = optimset('MaxIter', 50);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Minimize cost function (optimize)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ---------
% Visualize NN, after layer 1
%% ---------
fprintf('\nVisualizing Neural Network layer 1... \n')

figure
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ---------
% Predict using input X - check error
%% ---------

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Show predictions for displayed X - first 10
fprintf('\nPredictions vs Real outputs, for the 10 first from random selected X (shown in image): \n');

pred = predict(Theta1, Theta2, X(sel, :));

display([pred(1:10) y(sel)(1:10)]);