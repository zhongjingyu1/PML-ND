clc;
clear;
addpath(genpath('.'));
load('data/mirflickr.mat')
ttt=1;
params = [2^-6,    2^-2,   2^8,   2^0   0.1;     % 1.mirflickr
          2^-4,   2^-10,   2^-8,   2^0    1;     % 2.music_emotion
          2^0,  2^-10,   2^-10,  2^4   1000;     % 3.music_style
          2^-2,   2^10,   2^-4,   2^-10   1;     % 4.PML_emotions_2
          2^0,   2^10,   2^-4,   2^-6   0.1;     % 5.PML_emotions_3
          2^0,   2^10,  2^-10,  2^-10   0.1;     % 6.PML_emotions_4
          2^-4,   2^10,   2^0,    2^0   0.1;     % 7.PML_emotions_5
          2^2,   2^10,   2^-10,   2^0   0.1;     % 8.PML_medical_2
          2^4,   2^10,   2^-2,    2^0   0.1;     % 9.PML_medical_3
          2^4,   2^4,   2^-10,    2^0   100;     % 10.PML_medical_4
          2^2,   2^-4,   2^-10,   2^0   100;     % 11.PML_medical_5
          2^-10,   2^-2,  2^-10, 2^-10  0.1;     % 12.PML_health_5
          2^-8,   2^0,   2^-8,   2^-10  0.1;     % 13.PML_health_7
          2^-10,  2^-10, 2^-8,  2^-10   0.1;     % 14.PML_health_9          
          2^-10,   2^8,   2^-8,  2^-10  0.1;     % 15.PML_recreation_5
          2^-10,  2^8, 2^-10,   2^-2    0.1;     % 16.PML_recreation_7
          2^-10,  2^2, 2^-10,   2^-10   0.1;     % 17.PML_recreation_9
          2^-10,  2^10, 2^-10,   2^0    0.1;     % 18.PML_flags_4
          2^-2,  2^10, 2^-10,   2^-6    0.1;     % 19.PML_flags_5
          2^-10,  2^10, 2^-10,   2^0    0.1;     % 20.PML_education_5
          2^-4,  2^8, 2^-8,   2^-10     0.1;     % 21.PML_education_7
          2^-8,  2^0, 2^-10,   2^-6     0.1;     % 22.PML_education_9
          2^-10,  2^8, 2^-6,   2^-6     0.1;     % 23.PML_science_5
          2^-8,  2^8, 2^-10,   2^-10    0.1;     % 24.PML_science_7
          2^-8,  2^-8, 2^-8,   2^-10    0.1;     % 25.PML_science_9
          2^-8,  2^-8, 2^-8,   2^-10    0.1;     % 26.PML_genbase_2
          2^-8,  2^-8, 2^-8,   2^-10    0.1;     % 27.PML_genbase_3
          2^-8,  2^-8, 2^-8,   2^-10    0.1;     % 28.PML_genbase_4
          2^-8,  2^-8, 2^-8,   2^-10    0.1;     % 29.PML_genbase_5
          2^-10,  2^-8, 2^-6,   2^-10   0.1;     % 30.PML_Scene_2
          2^-10,  2^-8, 2^-6,   2^-10   0.1;     % 31.PML_Scene_3
          2^-10,  2^-8, 2^-6,   2^-10   0.1;     % 32.PML_Scene_4
          2^-6,  2^2, 2^-10,   2^-10    0.1;     % 33.PML_Scene_5
          2^-6,  2^2, 2^-10,   2^-10    0.1;     % 34.PML_Bibtex_4
          2^10,   2^2,  2^0,    2^2     0.1;     % 35.PML_Yeast_5
          2^10,    2^4,   2^2,    2^2   0.1;     % 36.PML_Yeast_7
          2^8,    2^0,   2^0,    2^0    100;     % 37.PML_Yeast_9
          2^-10,    2^4,   2^0,   2^2   100;     % 38.PML_art_5
          2^-6,    2^4,   2^-6,    2^-4   1;     % 39.PML_art_7
          2^-6,    2^0,  2^-10,  2^-8   0.1;     % 40.PML_art_9
          2^10,    2^6,     2^2,  2^2    10;     % 41.PML_Yeast_5
          2^10,    2^6,  2^0,   2^-10   100;     % 42.PML_Yeast_7
          2^10,   2^6,  2^-10,   2^-10   100;     % 43.PML_Yeast_9
          2^-2,  2^10, 2^-10,   2^-6    1;     % 44.PML_flags_6


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
    test_target = Truth_label(:, test_ind);

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
