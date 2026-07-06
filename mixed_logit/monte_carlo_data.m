
%{

This script generates a dataset using the mixed logit model.

%}

clear

rs = RandStream('twister', 'seed', 0);

n = 5000;
J = 5;
d = 4;

consumer_idx = repelem(1:n, J)';

rho = 0.8;    % correlation

X = sqrt(rho)*randn(rs, J*n, 1) + sqrt(1 - rho)*randn(rs, J*n, d);

X(:,3) = X(:,3) > quantile(X(:,3), 0.8);
X(:,4) = X(:,4) > quantile(X(:,4), 0.8);

X = zscore(X);

par_true = [-0.5, 0.5, 0.5, -0.5,   1.5, 0.5, 1.5, 0.5]';

Y = mix_logit_model(rs, par_true, X, consumer_idx);

save('data.mat', 'X', 'Y', 'consumer_idx', 'par_true')