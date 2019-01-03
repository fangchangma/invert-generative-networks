
function [loss, gradient] = evaluate_loss(z, net, A, G, y)
% function [loss] = evaluate_loss(z, net)
  z = single(z);

  % compute loss
  x = G(z);
  output = x(:);
  loss = 0.5 * norm(A * output - y)^2;

  % compute gradient  
  if nargout > 1
    dzdy = A'* (A * output - y);
    dzdy = single(reshape(dzdy, size(x)));
    res = vl_simplenn(net, z, dzdy);
    gradient = res.dzdx;
    gradient = double(gradient);
  end
end

