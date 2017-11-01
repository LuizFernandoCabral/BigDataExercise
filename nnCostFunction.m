function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];
         
% Return the following variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% Forward Propagation
% Layer 2
z_2 = Theta1 * X';
a_2 = sigmoid(z_2)';
% Add ones to the a_2 matrix
a_2 = [ones(size(a_2, 1), 1) a_2];

% Output Layer
z_3 = Theta2 * a_2';
a_3 = sigmoid(z_3);
h = a_3;

y_matrix = eye(num_labels)(y,:);

% Cost Function
J = sum(sum(-(y_matrix'.*log(h) + (1-y_matrix)'.*log(1-h))/m));

%% Regularization
temp1 = Theta1;
temp1(:,1) = 0;

temp2 = Theta2;
temp2(:,1) = 0;

reg1 = lambda*(sum(sum(temp1.^2)))/(2*m);
reg2 = lambda*(sum(sum(temp2.^2)))/(2*m);

J = J + reg1 + reg2;

%% Backward Propagation
d3 = a_3 - y_matrix';

d2 = Theta2(:,2:end)' * d3 .* sigmoidGradient(z_2);

delta_1 = d2 * X;

delta_2 = d3 * a_2;

Theta1_grad = delta_1/m;

Theta2_grad = delta_2/m;

reg1 = lambda*(temp1)/(m);
reg2 = lambda*(temp2)/(m);

%% Final regularized theta
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
