
%{

This function codes the training loop.
Based on the Matlab's built-in dlgradient and adamupdate.

%}

function [ema_net, train_pred, val_pred, test_pred] = ...
                learn(net, opt, nne, train_dataY, train_label, val_dataY, val_label, test_dataY)

num_iter = opt.num_iter;
max_step = opt.max_step;
B = opt.B;
print = opt.print;
% J = nne.J;
X = single(nne.X);

[p, L] = size(train_label);

input_cat_fcn  = @(dataY, X) cat(2, single(dataY), repmat(X, 1, 1, size(dataY, 3)));

learn_rate_fcn = @(t) max_step*min(10*t, 0.55 + 0.45*cos(pi*(t - 0.1)/0.9));
ema_update_fcn = @(F,f) F + 30/num_iter*(f - F);

val_input  = input_cat_fcn(val_dataY , X);
test_input = input_cat_fcn(test_dataY, X);

if opt.online
    train_label = gpuArray(train_label);
    train_dataY = gpuArray(train_dataY);
end
if opt.online == 2
    val_input = gpuArray(val_input);
end

weight = gpuArray(var(train_label, 0, 2).^-1);
weight = dlarray(weight, 'CB');
X = gpuArray(X);

train_pred = nan(p, L, 'single');
ema_net = net;
mtm1 = [];
mtm2 = [];

fcn = dlaccelerate(@loss_fcn);
clearCache(fcn)

if print; fprintf('<strong>  loss     |  val_loss  |  step    |  time    | iter </strong> \n'); end

tic

for iter = 1:num_iter + ceil(L/B)

    i = mod(iter*B + (-B:-1), L) + 1;

    label = gpuArray(train_label(:,i));
    dataY = gpuArray(train_dataY(:,:,i));   % transfer dataY to GPU before converting it

    input = input_cat_fcn(dataY, X);

    label = dlarray(label, 'CB');
    input = dlarray(input, 'SCB');

    if iter <= num_iter

        step = learn_rate_fcn(iter/num_iter);

        [loss, grad, state] = dlfeval(fcn, net, label, input, weight);
        [net, mtm1, mtm2] = adamupdate(net, grad, mtm1, mtm2, iter, step);
        net.State = state;

        ema_net = dlupdate(ema_update_fcn, ema_net, net.Learnables);

    end

    if ismember(iter, [1, round((1:10)/10*num_iter)])

        val_pred = minibatchpredict(net, val_input, inputData='SCB', outputData='CB');
        val_loss = mean(weight.*(val_pred - val_label).^2, 'all');
        if print; fprintf(' %9.2e | %9.2e  | %8.1e | %8.1e | %d \n', loss, val_loss, step, toc, iter); end
    end

    if iter == num_iter

        val_pred = minibatchpredict(ema_net, val_input, inputData='SCB', outputData='CB');
        val_loss = mean(weight.*(val_pred - val_label).^2, 'all');
        if print; fprintf(' %9.2e | %9.2e  | %8.1e | %8.1e | %d (EMA) \n', [], val_loss, [], toc, []); end
    end

    if iter > num_iter

        train_pred(:, i) = predict(ema_net, input, inputData='SCB', outputData='CB');
    end
end

test_pred = minibatchpredict(ema_net, test_input, inputData='SCB', outputData='CB');

end


%% -------------------------------------------------------------------------
%% loss function

function [loss, grad, state] = loss_fcn(net, label, input, weight)

[pred, state] = forward(net, input);

loss = mean(weight.*(pred - label).^2, 'all');
grad = dlgradient(loss, net.Learnables);

end