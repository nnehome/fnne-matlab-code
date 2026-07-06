
%{

This script trains the neural net.

%}

clearvars -except train val test nne rs print

%% settings

opt.max_step = 0.003;       % max step size in training
opt.num_iter = 3e4;         % number of training iterations
opt.B = 32;                 % mini batch size
opt.online = 1;             % GPU memory usage: 0: small(slow), 1: medium, 2: large(fast)
opt.print = print;          % display training information if print=true

if print; disp("Number of epochs to be run: " + opt.num_iter*opt.B/train.L); end

RandStream.setGlobalStream(rs)          % to seed neural net initialization
deep.gpu.deterministicAlgorithms(1);

%% layers

layers = [  
            inputLayer([size([nne.Y, nne.X]), NaN], 'SCB')

            convolution1dLayer(nne.J, 128, stride = nne.J)
            groupNormalizationLayer(128/4)
            geluLayer

            convolution1dLayer(1, 128)
            geluLayer

            convolution1dLayer(1, 128/2)
            globalAveragePooling1dLayer
            
            fullyConnectedLayer(128)
            geluLayer

            fullyConnectedLayer(numel(nne.names))
         ];

net = addLayers(dlnetwork, layers);
net = initialize(net);

%% train

[net, train.pred, val.pred, test.pred] = ...
    learn(net, opt, nne, train.dataY, train.label, val.dataY, val.label, test.dataY);

%% result

test_RMSE = mean((test.pred - test.label).^2, 2).^0.5;

estimate = predict(net, [nne.Y, nne.X], inputData='SCB', outputData='CB');

if print

    figure('position', [500 250 800 500])

    for k = 1:min(15, numel(nne.names))
        subplot(3, 5, k)
        scatter(test.label(k, :), test.pred(k, :), '.')
        refline(1, 0)
        xlabel(nne.names{k})
    end

    disp(" ")
    disp([table(test_RMSE, estimate, 'row', nne.names), nne.prior])

end

%% save 

nne.opt = opt;
nne.net = net;
save('trained_nne.mat', 'nne')

