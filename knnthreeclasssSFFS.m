clc
clear
close all
% groups: Normal=1 , Suspect=2 , Pathologic=3
data =xlsread ('data');
label=data(:,end);
Totaldata=data(:,1:22);
Totaldata=Totaldata';
label=label';
%% step1: devide data into train(70%) and test(30%)----k-fold cross validation(k=7)k=6
k=7;
fold=floor(size(Totaldata,2)/k);
c_all=0;
for i=1:k
    indtest= (i-1)*fold+1 : i*fold;
    indtrain= 1:size(Totaldata,2);
    indtrain(indtest)=[];
    datatrain= Totaldata(:,indtrain);
    dtrain= label(:,indtrain);
    
    datatest= Totaldata(:,indtest);
    dtest= label(:,indtest);
    %% Normalization
    mu= mean(datatrain,2);
    sigma= std(datatrain');
    for x=1:size(datatrain,2)
        datatrain(:,x)= (datatrain(:,x)-mu)./sigma';
    end
    for m=1:size(datatest,2)
        datatest(:,m)= (datatest(:,m)-mu)./sigma';
    end
    %% feature selection
    [sel,performance] = mySFFS_knn(datatrain,dtrain,datatest,dtest)
    % selection
    numf=1;
    datatrain= datatrain(sel(1:numf),:);
    datatest= datatest(sel(1:numf),:);
    %% step2&3: train classifier using datatrain
    mdl= fitcknn(datatrain',dtrain,'NumNeighbors',1,'Distance','Cityblock');
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
    
    c_all=c_all+C;
    
end
disp(['Total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['Accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])
%% plot confusion matrix
m= size(datatest,2);
c= unique(dtest);
v= numel(c);
one_h_output=zeros(3,m);
one_h_dtest=zeros(3,m);
dtest=(dtest)';
output= output';
for p=1:m
    one_h_output(output(p,1),p)=1;
    one_h_dtest(dtest(p,1),p)=1;
end
plotconfusion(one_h_dtest, one_h_output)
hold on
%% plot ROC
m= size(datatest,2);
c= unique(dtest);
v= numel(c);
one_h_output=zeros(3,m);
one_h_dtest=zeros(3,m);
dtest=dtest';
output= output';
for p=1:m
    one_h_output(output(1,p),p)=1;
    one_h_dtest(dtest(1,p),p)=1;
end
plotroc(one_h_dtest, one_h_output)