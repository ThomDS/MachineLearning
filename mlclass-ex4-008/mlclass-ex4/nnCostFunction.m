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
%D1 = zeros(size(Theta1)); % use to compute partial derivative
%D2 = zeros(size(Theta2)); % use of Capital D, but we can use also GradTheta1 & GradTheta2
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2)); 

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% adding bias unit for the first layer -- here a1 = a superscript one 
X = [ones(m,1) X]; 
a1 = X; 

z2 = Theta1*a1'; % size = m * number of units in layer 2
a2 = sigmoid(z2); 

% adding bias unit for the second layer
a2 = [ones(m,1) a2']; % the transpose if for having a matrix dim = 5000 * 26

z3 = Theta2*a2'; 
a3 = sigmoid(z3); % a3 = h_theta(x), dim = 10*5000

% Now we need to recode y to have vector of values 0 or 1

yVec = zeros(num_labels, m); 
for i=1:m
	yVec(y(i), i) = 1; % so that we replace value 0 by value 1 
end 

% Compute Cost following the formula

J = 1/m * sum(sum((-yVec) .* log(a3) - (1 - yVec) .* log(1-a3)));

% Adding Regularization term
J = J + lambda/(2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) + ...
    sum(sum(Theta2(:,2:end) .^ 2)))


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for i=1:m % for all training example

	% set a1 = X
	a1 = X(i,:); % X with bias unit, dim = 1*401
	
	% Perform forward propagation to compute a^(l), l=2,..., L (here L = 3)
	z2 = Theta1*a1'; % (25*401) x (401*1) = (25*1)
	a2 = sigmoid(z2); 
	a2 = [1; a2]; % adding bias = (26,1)
	z3 = Theta2*a2; % (10*26) x (26, 1) = (10*1)
	a3 = sigmoid(z3); 
	
	% Using y^(i) compute delta^(L)
	delta3 = a3-yVec(:,i);
	
	% compute delta^(l), l=2,...,L-1 (here just compute delta2)
	% NB : remove the column with bias unit ... 
	delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2); % (25*10) x (10*1) .x (25*1)
	
	% Now increment D1 & D2
	
	Theta1_grad = Theta1_grad + delta2*a1; 
	Theta2_grad = Theta2_grad + delta3*a2'; 
	
endfor

Theta1_grad = (1/m).*Theta1_grad ; % without regularization term
Theta2_grad = (1/m).*Theta2_grad ; 


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end); 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end); 
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
