
function W = create_weights(kernel_size, num_out_filters, num_in_filters, type)
  if strcmp(type,'[0,1]')
    W = rand(kernel_size,kernel_size,num_out_filters,num_in_filters,'single');
  elseif strcmp(type,'gaussian')
    W = 1/sqrt(kernel_size^2*num_out_filters) * randn(kernel_size,kernel_size,num_out_filters,num_in_filters,'single');
  elseif strcmp(type,'{0,1}')
    W = rand(kernel_size,kernel_size,num_out_filters,num_in_filters,'single') > 0.5;
  elseif strcmp(type,'{-1,1}')
    W = 2 * (rand(kernel_size,kernel_size,num_out_filters,num_in_filters,'single')>0.5) - 1;
  elseif strcmp(type,'[-1,1]')
    W = 2 * rand(kernel_size,kernel_size,num_out_filters,num_in_filters,'single') - 1;
  elseif strcmp(type,'1')
    W = ones(kernel_size,kernel_size,num_out_filters,num_in_filters,'single');
  else
    error('invalid type')
  end
  W = single(W);
end