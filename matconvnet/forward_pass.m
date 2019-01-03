function [out] = forward_pass(z, net)
  % run the CNN
  z = single(z);  
  res = vl_simplenn(net, z) ;
  out = res(end).x;
  out = double(out);
end

