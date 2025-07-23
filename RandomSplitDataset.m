function [train_data, test_data, train_p_target, test_target] = RandomSplitDataset(dataset, train_ratio)

load(dataset);

randIndex = randperm(size(data, 1));
trainIndex = randIndex(randIndex(1:ceil(train_ratio*size(data, 1))));
testIndex = setdiff(randIndex, trainIndex);

train_data = data(trainIndex, :);
%train_p_target = partial_target(:, trainIndex)';
train_p_target = target(:, trainIndex)';
test_data = data(testIndex, :);
test_target = target(:, testIndex)';

end