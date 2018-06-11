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
       
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2=a2';
n=size(a2,1);
a2=[ones(n, 1) a2];
z3 = Theta2*a2';
a3=sigmoid(z3);
a3=a3';
[min_values indices]=max(a3, [], 2);
p=indices;
for i=1:m
  c_y=zeros(num_labels,1);
  c_y(y(i))=1;
    J=J+ (-(c_y)'*log(a3(i,:))'-(1-c_y)'*log(1-a3(i,:))')/m;
end
temp_theta1=sum(sumsq(Theta1)')-sumsq(Theta1(:,1));
temp_theta2=sum(sumsq(Theta2)')-sumsq(Theta2(:,1));
%total=(sum(temp_theta1)-sum(temp_theta1(:,1))+sum(temp_theta2)-sum(temp_theta2(:,1)));
J=J+(temp_theta1+temp_theta2)*lambda/(2*m);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
del1=zeros(size(Theta1,1),size(Theta1,2));
del2=zeros(size(Theta2,1),size(Theta2,2));

for t=1:m
z2 = Theta1*(X(t,:))';
a2 = sigmoid(z2);
a2=a2';
n=size(a2,1);
a2=[ones(n, 1) a2];
z3 = Theta2*a2';
a3=sigmoid(z3);
a3=a3';
  c_y=zeros(num_labels,1);
  c_y(y(t))=1;
  d3 = a3'-c_y;
  
  %fprintf("size = %d",size(d3));
  
  d2 = ((Theta2)'*d3);
  d2 = d2(2:end,:).*sigmoidGradient(z2);
  %fprintf("size = %d",size(d2));
  del1=del1+d2*(X(t,:));
  del2=del2+d3*(a2);
end
Theta1_grad=del1/m;
Theta2_grad=del2/m;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
