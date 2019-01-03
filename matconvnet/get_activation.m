function nonlinear = get_activation(activation)
%   display(activation)
  if strcmp(activation, 'relu')
    nonlinear = struct('type', 'relu');
  elseif strcmp(activation, 'leakyrelu')
    nonlinear = struct('type', 'relu', 'leak', 0.2);
  elseif strcmp(activation, 'sigmoid')
    nonlinear = struct('type', 'sigmoid');
  elseif strcmp(activation, 'tanh')
    nonlinear = struct('type', 'tanh');
  end
end
