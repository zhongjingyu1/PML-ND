clc;
clear;
addpath(genpath('.'));
load('data/mirflickr.mat')
ttt=1;
params = [2^-6,    2^-2,   2^8,   2^0   0.1;     % 1.mirflickr
];
partial_labels=candidate_labels;
beta=params(ttt,1);
lambda=params(ttt,2);
delta=params(ttt,3);
gamma=params(ttt,4);
enaf=params(ttt,5);

nfold = 10;                 %ten fold crossvalidation
k=10;
[n_sample,~]= size(data);
result=zeros(nfold,4); %save evaluation result
n_test = round(n_sample/nfold);

I = 1:n_sample;
[Truth_label] =Truth_label_export(data,partial_labels,k);
Truth_label=Truth_label';
for i=1:nfold%
    fprintf('data2 processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if start_ind+n_test-1 > n_sample
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    train_data = data(train_ind, :);
    train_p_target = partial_labels(:,train_ind);
    Truth_label1 = Truth_label(:,train_ind);
    test_data = data(test_ind,:);
    test_target = target(:, test_ind);

    tic
    [W,obj] =PML_ND(train_data,train_p_target',Truth_label1',beta,lambda,gamma,delta,enaf);
    tt=toc;

    [pre_labels, pre_dis , res_once] = PML_ND_predict(W, test_data, test_target);
    result(i,:)=res_once;
end
rr=sum(result)/nfold
rr2=std(result)

A=(Truth_label==target);
ground_truth_labels_predict_peicison=sum(sum(A))/(size(A,1)*size(A,2))
