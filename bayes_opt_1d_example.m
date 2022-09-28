% Demonstration of Bayesian optimization in MATLAB
%
% This code has been tested with MATLAB version 2021b
%

% Function to maximise (scalar)
f_to_max = @(x) -0.05 * x.^2 - cos(x) + 0.25 * sin(3 * x + 0.8) + 5;

% Plot function
X = linspace(-5, 5, 100);
Y = f_to_max(X);

figure(1); clf
plot(X, Y); hold on
xlabel('$x$', 'Interpreter', 'latex')
ylabel('$f(x)$', 'Interpreter', 'latex')
grid on

% Try to find minimum by optimization
% (finds a local minimum)
FUN = @(x) 1 / f_to_max(x);
x0 = 2;
LB = -5;
UB = 5;
x_max = fmincon(FUN,x0,[],[],[],[],LB,UB);

% Add max point to plot
plot(x_max, f_to_max(x_max), 'o')
text(x_max+0.08, f_to_max(x_max)+0.08, sprintf("Max: (%.3f, %.3f)", x_max, f_to_max(x_max)))

% Define bounded region of parameter space
pbounds = [-5, 5];
x = optimizableVariable('x', pbounds, 'Type', 'real');
results = bayesopt(@(x) 1 ./ f_to_max(x{1, 'x'}), x);
x_max_BO = results.XAtMinObjective{:, "x"};
f_max_BO = 1 / results.MinObjective;

fprintf("Maximum from Bayesian optimization: (%.3f, %.3f)\n", ...
    x_max_BO, f_max_BO)

