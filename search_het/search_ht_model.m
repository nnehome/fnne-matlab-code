
%{

This function codes the search model with unobserved heterogeneity.
The first input 'rs' is a random number stream (RandStream).

%}

function [y, stat] = search_ht_model(rs, par, curve, X, consumer_idx)

n = consumer_idx(end);
J = height(X)/n;
d = width(X);

par = par(:)';

alfa     = par( 1       );      % search cost average
eta      = par( 2       );      % outside utility average
sigma_c  = par( 3       );      % search cost heterogeneity
sigma_u  = par( 4       );      % outside utility heterogeneity
sigma_xi = par( 5       );      % pre-search shock size
beta     = par( 6 : d+5 );      % coefficients for product attributes

%% 3D array

X = reshape(X, J, n, d);

%% primitives

coef  = reshape(beta, 1, 1, d);

v = - eta + sum(X.*coef, 3) + randn(rs, 1, n, 1) *sigma_u + randn(rs, J, n, 1)*sigma_xi;
u = v + randn(rs, J, n, 1);

c = alfa + randn(rs, 1, n, 1)*sigma_c;
r = v + interpolate(c, curve.log_cost, curve.utility);

%% model calculation

[r, i] = sort(r, 'descend');
i = i + (0:n-1)*J;

u = u(i);

searched = cummax([zeros(1,n); u(1:J-1,:)]) <= r;
searched(1,:) = true;

first = (1:J)' == ones(1, n);
last  = (1:J)' == sum(searched);

u( ~ searched) = - inf;
bought = u == max(0, max(u));

searched(i) = searched;
bought(i) = bought;
first(i) = first;
last(i) = last;

y = [searched(:), bought(:), first(:), last(:)];

%% summary statistics

srh_num  = mean( sum(searched));
srh_rate = mean( sum(searched) > 1);
buy_rate = mean( sum(bought) > 0);

stat = [srh_num, srh_rate, buy_rate];

end

%% sub-function: interpolation

function out = interpolate(query, x, v)

    % x needs to be sorted from smallest to largest
    % x may NOT contain duplicates.
    % each query point must be in [x(1), x(end)].
    
    q = query(:);

    [~,j] = max( q <= x', [], 2);
    j = max(2,j);
    
    xl = x(j-1);
    xr = x(j);
    vl = v(j-1);
    vr = v(j);

    out = vr - (xr - q)./(xr - xl).*(vr - vl);

    out = reshape(out, size(query));

end
