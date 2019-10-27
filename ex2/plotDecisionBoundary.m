function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    % You're using x2 for the x axis. Remember x1 is all ones.
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    % Remember the decision boundary is z=0 in g(z).
    % g(z) = sigmoid(z) and z = theta * x where theta * x = theta_1 +
    % theta_2*x2 + theta_3*x3.
    % You're using x2 on the X axis, so solve for x3.
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    
    % If there are more than two columns in X the function assumes that mapFeature was used to perform polynomial regression using
    % X1 and X2 as a base.
    % The first part of the code simply sets up a grid of values to use for a contour plot.
    
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    % Then mapFeature is called to set up the features for polynomial regression on each point of the grid.
    % Those features are matrix multiplied by theta to compute a z value on each grid point.
    % z is essentially the logistic hypothesis without the sigmoid call.
    
    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    % Remember theta will have more than 3 columns (features)!
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % The decision boundary is where the z value is zero, that is, where the hypothesis would be the 0.5 threshold after the sigmoid is called.
    % To plot that boundary a level argument of [0,0] is added to the contour call.
    
    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
