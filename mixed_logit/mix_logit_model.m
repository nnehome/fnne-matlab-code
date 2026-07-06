
%{

This function codes the mixed logit model.

%}

function Y = mix_logit_model(rs, par, X, consumer_idx)

n = consumer_idx(end);
d = width(X);
J = height(X)/n;

par = par(:)';

mu    = par( 1:d     );
sigma = par( d+1:2*d );

X = reshape(X, J, n, d);    % J x n x d

coef = reshape(mu, 1, 1, d) + randn(rs, 1, n, d).*reshape(sigma, 1, 1, d);    % 1 x n x d

v = sum(X.*coef, 3);        % J x n
e = exp(v);                 % J x n

p = e./sum(e, 1);           % J x n
f = cumsum(p, 1);           % J x n

draw = rand(rs, 1, n);
Y = draw >= f - p & draw < f;

Y = reshape(Y, n*J, 1);