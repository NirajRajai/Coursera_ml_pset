function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h=X*theta;
sum=0;
for i=2:size(theta)
  sum+=theta(i)*theta(i);
  end
J=((h-y)'*(h-y))/(2*m) + (lambda*sum)/(2*m);
temp=X(:,1);
grad(1)=(temp'*(h-y))/m;
for i=2:size(theta)
  temp=X(:,i);
  grad(i)=(temp'*(h-y))/m +(lambda*theta(i))/m;
  end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
