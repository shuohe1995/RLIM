%parametets
lambda=0.1;
beta1=0.1;
beta2=0.1;
k = 10 ;
max_iter = 50; 
ker = 'rbf';
times=10;
tol=1e-5;
%
ranking_loss=zeros(1,times);
coverage=zeros(1,times);
one_error=zeros(1,times);
average_precision=zeros(1,times);
hamming_loss=zeros(1,times);
Mi_F1=zeros(1,times);
Ma_F1=zeros(1,times);
%
dataname='Image.mat';
%
for t=1:times
fprintf("---------------The %d-th time-------------\n",t);
[train_data, test_data, train_p_target, test_target] = RandomSplitDataset(dataname,0.5);%
train_data=sparse(train_data);
test_data=sparse(test_data);
[m,d]=size(train_data);
[m,l]=size(test_target);
par = mean(pdist(train_data));
%
%
[S] = ConstructSimilarityMatrix(train_data, k);
[C] = ConstructLabelSimilarityMatrix(train_p_target);
%initial P
P = train_p_target;
%
[H, A,b,Aeq, beq, lb, ub, opts] = QPSettings(S,C,train_p_target,beta1,beta2);
%
i=1;
tol1=1;
train_outputs=P;
test_outputs=sparse(m,l);
while i<=max_iter && tol1>tol
    P1=P;
    t1=train_outputs;
    [train_outputs, test_outputs] = MulRegression(train_data, P, test_data, lambda, par, ker);
    [P,fval] = QP(train_outputs, H, A,b,Aeq, beq, lb, ub, opts);
    P2=P;
    t2=train_outputs;
    tol1=norm(P2-P1, 'fro' );
    i=i+1;
end
%
%
test_outputs=test_outputs';
test_target=test_target';
%
ranking_loss(t)=Ranking_loss(test_outputs, test_target);
coverage(t)=Coverage(test_outputs, test_target);
one_error(t)=One_error(test_outputs, test_target);
average_precision(t)=Average_precision(test_outputs, test_target);
%
%
[l,m]=size(test_outputs);
pre_labels=zeros(l,m);
pre_labels(find(test_outputs<0))=-1;
pre_labels(find(test_outputs>0))=1;
%
hamming_loss(t)=Hamming_loss(pre_labels, test_target);
Mi_F1(t)=MicroF1(pre_labels, test_target);
Ma_F1(t)=MacroF1(pre_labels, test_target);
pre_labels=pre_labels';
test_target=test_target';
%
end
fprintf("one_error:%0.3f(%0.3f)\r\n",mean(one_error),std(one_error));
fprintf("HammingLoss:%0.3f(%0.3f)\r\n",mean(hamming_loss),std(hamming_loss));
fprintf("ranking_loss:%0.3f(%0.3f)\r\n",mean(ranking_loss),std(ranking_loss));
fprintf("coverage:%0.3f(%0.3f)\r\n",mean(coverage),std(coverage));
fprintf("average_precision:%0.3f(%0.3f)\r\n",mean(average_precision),std(average_precision));
fprintf("MacroF1:%0.3f(%0.3f)\r\n",mean(Ma_F1),std(Ma_F1));
fprintf("MicroF1:%0.3f(%0.3f)\r\n",mean(Mi_F1),std(Mi_F1));


