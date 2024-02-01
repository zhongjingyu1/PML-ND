function [predLabel] = Truth_label_export(data,partial_labels,k)
partial_labels=partial_labels';
[~,label_num] = size(partial_labels);
fea_num = size(data,2);
ins_num = size(data,1);

[~,neighbor] = pdist2(data,data,'euclidean','Smallest',k+1);
neighbor=neighbor';
neighbor=neighbor(:,2:k+1);
datas = zeros(ins_num,k);
trans = zeros(ins_num);
rows = repmat((1:ins_num)',1,k);
for i=1:ins_num
    neighborIns = data(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,data(i,:)');
    datas(i,:) = w;
end
trans = sparse(rows,neighbor,datas,ins_num,ins_num);
sumW = full(sum(trans,2));
sumW(sumW==0)=1;
trans = bsxfun(@rdivide,trans,sumW);

partial_labels_1=abs(partial_labels-1);
mu1=0.5;
R = pdist2( partial_labels_1'+eps, partial_labels_1'+eps, 'cosine' );
A=data'*data+mu1*ones(fea_num,fea_num);
B=mu1*R;
C=data'*partial_labels_1;
W_1 = lyap(A, B, C);  
for i=1:ins_num
    K(i,:)=data(i,:)*W_1;
end
P1=K;
for i=1:ins_num
    P1(i,:)=(P1(i,:)-min(min(P1(i,:))))/(max(max(P1(i,:)))-min(min(P1(i,:)))+eps);
end
P1=P1.*partial_labels;
K_1 = P1./(sum(P1,2)+eps);

D=zeros(ins_num,label_num);
af=0.999;
for i=1:ins_num
    for j=1:label_num
        if partial_labels(i,j)==1
            D(i,j)=K_1(i,j);
        else
            D(i,j)=af;
        end
    end
end
P=partial_labels;
p0=P;
iterVal = zeros(1,100);
for iter=1:100
    tmp= P;
    r=trans*P;
    P = (1-D).*r+D.*p0;
    P = P.*partial_labels;                              
    P = P./(repmat(sum(P,2),1,label_num)+eps);
    diff=norm(full(tmp)-full(P),2);
    iterVal(iter) = abs(diff);
    if abs(diff)<0.01
        break
    end
end
predLabel = zeros(ins_num,label_num);
for i=1:ins_num
    [val,idx] = max(P(i,:));
    predLabel(i,idx)=1;
end

for i=1:ins_num
    for j=1:label_num
        if P(i,j)>0.25
            predLabel(i,j)=1;
        end
    end
end