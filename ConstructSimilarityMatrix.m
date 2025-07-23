function [S] = ConstructSimilarityMatrix(train_data, k)

dist_matrix = pdist2(train_data, train_data, 'Euclidean');
[sorted_dist, sorted_index] = sort(dist_matrix);
sorted_dist = sorted_dist(2:(k+1), :);
sorted_index = sorted_index(2:(k+1), :);
sorted_dist2 = sorted_dist.^2;
sigma = mean(sorted_dist(k,:));

S = exp(-sorted_dist2/(sigma^2));
m = size(train_data, 1);

W = zeros(m, m);

for i = 1:m
   W(sorted_index(:,i), i) = S(:,i); 
end

S = W + W';

end