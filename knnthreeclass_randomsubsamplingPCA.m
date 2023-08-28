clc
clear
close all
% groups: Normal=1 , Suspect=2 , Pathologic=3
data =xlsread ('data');
label=data(:,end);
Totaldata=data(:,1:21);
Totaldata=Totaldata';
label=label';
%%  devide data into train(70%) and test(30%)----random subsampling
div= 0.7;
num= round( div * size(Totaldata,2));
for i=1:100
    ind= randperm(size(Totaldata,2));
    Totaldata= Totaldata(:,ind);
    label= label(ind);
    %
    datatrain=Totaldata(:,1:num);
    dtrain= label(1:num);
    
    datatest=Totaldata(:,num+1:end);
    dtest=label(num+1:end);
    %% Normalization
    mu= mean(datatrain,2);
    sigma= std(datatrain');
    for x=1:size(datatrain,2)
        datatrain(:,x)= (datatrain(:,x)-mu)./sigma';
    end
    for m=1:size(datatest,2)
        datatest(:,m)= (datatest(:,m)-mu)./sigma';
    end
    %% Dimension reduction---PCA
    m=3;
    C= cov(datatrain');
    [U,D]= eig(C);
    D= diag(D);
    [D,ind]= sort(D,'descend');
    U= U(:,ind);
    W= U(:,1:m);
    datatrain= W'*datatrain;
    datatest=  W'*datatest;
    
    %% step2 & 3
    mdl= fitcknn(datatrain',dtrain,'NumNeighbors',1)
    output= predict(mdl, datatest')';
    %% Confusion matrix
    C= confusionmat(dtest,output)
    % %% total accuracy
    accuracy(i)= sum(diag(C)) / sum(C(:))*100;
    
    %% accuracy 1
    accuracy1(i)= sum(C(1,1)) / sum(C(1,:))*100;
    
    %%  accuracy 2
    accuracy2(i)= sum(C(2,2)) / sum(C(2,:))*100;
    
    %%  accuracy 3
    accuracy3(i)= sum(C(3,3)) / sum(C(3,:))*100;
    i
end
disp(['Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['Accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])
