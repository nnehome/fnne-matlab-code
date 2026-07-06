
%{

This script generates the training, validation, and test examples.

%}

clear
print = true;
load('data.mat', 'Y', 'X', 'consumer_idx')
rs = RandStream.create('threefry', 'seed', 1, 'stream', 2, 'nums', 2);

%% data dimension

[~, i] = sortrows([consumer_idx, X]);
X = X(i,:);
Y = Y(i,:);

n = consumer_idx(end);
J = height(X)/n;
d = width(X);

%% setup

para = { 
         "\mu_"+(1:d)',      [-1,  1].*ones(d,1)
         "\sigma_"+(1:d)',   [ 0,  2].*ones(d,1)
         };

names = cellstr(vertcat(para{:,1}));
prior = array2table(vertcat(para{:,2}), var = {'l','u'});

if print; fprintf('n = %d, J = %d, d = %d, ', n, J, d); end
if print; fprintf('Number of parameters = %d \n', height(prior)); end

%% generate examples

train.L = 5e4;
val.L   = 1e3;
test.L  = 1e3;

tic

[train.label, train.dataY, train.stats] = generate(rs, train.L, prior, X, consumer_idx);
[val.label,   val.dataY,   ~          ] = generate(rs, val.L,   prior, X, consumer_idx);
[test.label,  test.dataY,  ~          ] = generate(rs, test.L,  prior, X, consumer_idx);

if print; toc; end

%% statistics distribution

% n/a

%% update nne structure

nne.names = names;
nne.prior = prior;

nne.J = J;
nne.Y = Y;
nne.X = X;
nne.consumer_idx = consumer_idx;


%% ========================================================================

function [label, dataY, stats] = generate(rs, L, prior, X, consumer_idx)

n = consumer_idx(end);
J = height(X)/consumer_idx(end);
p = height(prior);

label = zeros(p, L, 'single');
dataY = zeros(n*J, 1, L, 'single');
stats = nan(0, L, 'single');

pts = rand(rs, p, L);

c = 1.5;    % tr-normal(0,1,-c,+c) scaled to [0,1].
pts = 0.5 + (0.5/c)*norminv((1 - pts)*normcdf(-c) + pts*normcdf(c));

parfor l = 1:L

    par = prior.l + (prior.u - prior.l).*pts(:,l);

    rss = RandStream.create(rs.Type, 'seed', rs.Seed, 'stream', rs.StreamIndex, 'nums', rs.NumStreams);
    rss.Substream = rs.Substream + l;
    
    Y = mix_logit_model(rss, par, X, consumer_idx);

    dataY(:, :, l) = Y;
    label(:, l) = par;
    % stats(: ,l) = stat;

end

rs.Substream = rs.Substream + L + 1;  % important

end