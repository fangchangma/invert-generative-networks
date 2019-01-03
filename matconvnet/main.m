close all; clear; clc;

% Dimension of the latent space. Set k=2 for visualization purposes
k = 2;
perc_sample = 0.1; % percentage of samples

% X is a SINGLE array of dimension H x W x D x N where 
%   (H,W) are the height and width of the image stack, 
%   D is the image depth (number of feature channels), 
%   N the number of of images in the stack.
z_gt = single(randn(1,1,k)); % dont forget 'single'

weight_type = 'gaussian'; % 'gaussian', '{0,1}', '{-1,1}', '[-1,1]', '1'
measurement_type = 'subsample'; % 'subsample', 'gaussian'
activation = 'relu'; % 'relu', 'leakyrelu', 'sigmoid', 'tanh'

%% Define network 
net.layers = {} ;
kernel_size = 3;
stride = 2;
filters = [k, 4, 8, 16, 1];
num_out_channels = filters(end);
% F is a SINGLE array of dimension FW x FH x K x FD where 
%   (FH,FW) are the filter height and width, 
%   K the number of filters in the bank, 
%   FD the depth of a filter (the same as the depth of image X).
for i = 2 : length(filters)
    net.layers{end+1} = struct(...
        'type', 'convt', ...
        'weights', {
            {create_weights(kernel_size,filters(i),filters(i-1),weight_type), ...
            zeros(1, 1, filters(i), 'single') ...
        }}, ...
        'upsample', stride);
    net.layers{end+1} = get_activation(activation);
end
net = vl_simplenn_tidy(net);


%% Generative model
G = @(z)forward_pass(z, net);
x_gt = G(z_gt);
fprintf('The convolutional neural network has %d layers \n', size(filters, 2)-1);
fprintf('The input size is %d x %d (%d channels), with a total of %d pixels \n', ...
    size(z_gt, 1), size(z_gt, 2), k, numel(z_gt));
fprintf('The final output size is %d x %d (%d channels), with a total of %d pixels \n', ...
    size(x_gt, 1), size(x_gt, 2), num_out_channels, numel(x_gt));
x_gt = x_gt(:);

%% Sampling
num_out = numel(x_gt);
m = floor(num_out * perc_sample); 
fprintf('The measured vector has %d pixels (%s)\n', m, measurement_type);

if strcmp(measurement_type, 'gaussian')
  % Gaussian Measurements: each measurement is a linear combination of all
  % pixels
  A = create_weights(num_out, m, 'gaussian'); 
elseif strcmp(measurement_type, 'subsample')
  % Subsample Measurements: only a small subset of pixels is observed
  A = eye(m, num_out); 
else
  error('invalid measurement_type')
end
y = A * x_gt;

%% Loss function
C = @(z)(evaluate_loss(z, net, A, G, y));

%% Create random initial estimates
z0 = randn(1,1,k);     % Random initializer

%% Optimization 
options = optimoptions('fminunc','Display', 'off', 'SpecifyObjectiveGradient', true);

t = cputime;
z_hat_1 = fminunc(C, z0, options);
z_hat_2 = fminunc(C, -z_hat_1, options);
t = cputime - t;

if C(z_hat_1) < C(z_hat_2)
  z_hat = z_hat_1;
else
  z_hat = z_hat_2;
end
x_hat = G(z_hat);


%% Evaluation
diff_z = norm2(z_hat-z_gt);
diff_x = norm2(x_hat(:)-x_gt);
diff_C = C(z_hat)-C(z_gt);
fprintf('|z_hat - z_gt| = %g \n', diff_z)
fprintf('|x_hat - x_gt| = %g \n', diff_x)
fprintf('|C_hat - C_gt| = %g \n', diff_C)
error_z = 100 * diff_z / norm2(z_gt); % percentage of error
fprintf('Relative estimation error in latent space: %2.2f%% \n', error_z)

if (k==2) 
    markersize = 40;
    fontsize = 20;
    gridsize = 4.5;
    boundary = -8.0 : 0.25 : 8;

    [XX,YY] = meshgrid(boundary,boundary);
    XX = XX + z_gt(1);
    YY = YY + z_gt(2);
    P = [XX(:),YY(:)];
    Cs = zeros(size(P,1), 1);
    for i = 1 : size(P,1)
      z_i = reshape(P(i,:)', [1, 1, k]);
      Cs(i) = C(z_i);
    end
    Cs = reshape(Cs, size(XX));

    fig1 = figure(1);
    %     set(gca,'position',[0 0 1 1],'units','normalized')
    cla;
    mesh(XX, YY, Cs);
    hold on
    contour(XX, YY, Cs, 'LevelStep', max(Cs(:))/100 )
    colormap('jet')
    hold on;
    z_offset=max(Cs(:))/100;
    p1 = scatter3(z_hat(1), z_hat(2), C(z_hat)+z_offset, 2*markersize, 'or', ...
      'MarkerFaceColor', 'r');
    p2 = scatter3(z_gt(1), z_gt(2), C(z_gt)+z_offset, markersize, '*g', ...
      'MarkerFaceColor', 'g');
    title('Cost function landscape')
    legend([p1, p2], 'z_{hat}', 'z_{gt}')
    xlabel('z(1)')
    ylabel('z(2)')
else
    disp('Visualization is available only for 2-dimensional latent space.')
end

function n = norm2(A) 
  n = norm(squeeze(A));
end

