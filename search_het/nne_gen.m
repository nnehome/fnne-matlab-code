
%{

This script generates the training, validation, and test examples.

%}

clear
print = true;
load('data.mat', 'X', 'Y', 'consumer_idx')
rs = RandStream.create('threefry', 'seed', 1, 'stream', 2, 'nums', 2);

%% prepare data

[~, i] = sortrows([consumer_idx, X]);   % ensure data not sorted by outcomes
X = X(i,:);
Y = Y(i,:);

n = consumer_idx(end);
J = height(X)/n;
d = width(X);

%% setup

para = { "\alpha",         [-5, -2]                % search cost
         "\eta",           [ 3,  6]                % outside utility
         "\sigma_c",       [ 0,  2]                % search cost heterogeneity
         "\sigma_u"',      [ 0,  2]                % outside utility heterogeneity
         "\sigma_\xi",     [ 1,  3]                % pre-search shock size
         "\beta_"+(1:d)',  [-0.5,  0.5].*ones(d,1)
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

if print; fprintf(' Search num  range: [%1.2f  %1.2f] \n', prctile(train.stats(1,:), [1, 99])); end
if print; fprintf(' Search rate range: [%1.3f  %1.3f] \n', prctile(train.stats(2,:), [1, 99])); end
if print; fprintf(' Buy rate range:    [%1.3f  %1.3f] \n', prctile(train.stats(3,:), [1, 99])); end

%% update nne structure

nne.names = names;
nne.prior = prior;

nne.J = J;
nne.Y = Y;
nne.X = X;
nne.consumer_idx = consumer_idx;


%% ========================================================================

function [label, dataY, stats] = generate(rs, L, prior, X, consumer_idx)

load('curve.mat', 'curve')

n = consumer_idx(end);
J = height(X)/consumer_idx(end);
p = height(prior);

label = zeros(p, L, 'single');
dataY = zeros(n, 4, L, 'uint64');
stats = nan(3, L, 'single');

pts = rand(rs, p, L);

c = 1.5;      % tr-normal on [-c,+c] scaled to [0,1].
pts = 0.5 + (0.5/c)*norminv((1 - pts)*normcdf(-c) + pts*normcdf(c));

parfor l = 1:L

    par = prior.l + (prior.u - prior.l).*pts(:,l);

    rss = RandStream.create(rs.Type, 'seed', rs.Seed, 'stream', rs.StreamIndex, 'nums', rs.NumStreams);
    rss.Substream = rs.Substream + l;
    
    [Y, stat] = search_ht_model(rss, par, curve, X, consumer_idx);

    dataY(:, :, l) = uint64(bit2int(Y, J));
    label(:, l) = par;
    stats(: ,l) = stat;

end

rs.Substream = rs.Substream + L + 1;  % important

end
