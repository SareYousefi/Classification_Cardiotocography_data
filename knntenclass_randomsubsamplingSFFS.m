clc
clear
close all
% groups: 10
data =xlsread ('dataclass');
label=data(:,end);
Totaldata=data(:,1:21);
Totaldata=Totaldata';
label=label';

%% step1: devide data into train and test----random subsampling
div= 0.7;
num= round( div * size(Totaldata,2));
for i=1:5
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

%% feature selection
    [sel,Performance] = mySFFS_knnten(datatrain,dtrain,datatest,dtest)
%     [~,bestsel]= sort(Performance,'descend');
%     numf= bestsel(1);
    % selection
    numf=9;
    datatrain= datatrain(sel(1:numf),:);
    datatest= datatest(sel(1:numf),:);
    %% step2: train classifier using datatrain
    mdl= fitcknn(datatrain',dtrain,'NumNeighbors',1,'Distance','Cityblock','Standardize',1);
% behtarin tedad split shod ba accuracy 57%
%     mdl= fitctree(datatrain',dtrain,'MaxNumSplits',70);
    %% step3: test trained classifier
    
        output= predict(mdl, datatest')';
    %% Confusion matrix
    C= confusionmat(dtest,output)
    % %% total accuracy
    accuracy(i)= sum(diag(C)) / sum(C(:))*100;
    
    % %% accuracy 1
    accuracy1(i)= sum(C(1,1)) / sum(C(1,:))*100;
    
    %%  accuracy 2
    accuracy2(i)= sum(C(2,2)) / sum(C(2,:))*100;
    
    %%  accuracy 3
    accuracy3(i)= sum(C(3,3)) / sum(C(3,:))*100;
    
    %%  accuracy 4
    accuracy4(i)= sum(C(4,4)) / sum(C(4,:))*100;
    
    %%  accuracy 5
    accuracy5(i)= sum(C(5,5)) / sum(C(5,:))*100;
    
    %%  accuracy 6
    accuracy6(i)= sum(C(6,6)) / sum(C(6,:))*100;
    
    %%  accuracy 7
    accuracy7(i)= sum(C(7,7)) / sum(C(7,:))*100;
    
    %%  accuracy 8
    accuracy8(i)= sum(C(8,8)) / sum(C(8,:))*100;
    
    %%  accuracy 9
    accuracy9(i)= sum(C(9,9)) / sum(C(9,:))*100;
    
    %%  accuracy 10
    accuracy10(i)= sum(C(10,10)) / sum(C(10,:))*100;
    
    i
end
disp(['Total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['Accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['Accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['Accuracy3: ',num2str(mean(accuracy3)) ,'%'])
disp(['Accuracy4: ',num2str(mean(accuracy4)) ,'%'])
disp(['Accuracy5: ',num2str(mean(accuracy5)) ,'%'])
disp(['Accuracy6: ',num2str(mean(accuracy6)) ,'%'])
disp(['Accuracy7: ',num2str(mean(accuracy7)) ,'%'])
disp(['Accuracy8: ',num2str(mean(accuracy8)) ,'%'])
disp(['Accuracy9: ',num2str(mean(accuracy9)) ,'%'])
disp(['Accuracy10: ',num2str(mean(accuracy10)) ,'%'])
%% plot confusion matrix
m= size(datatest,2);
c= unique(dtest);
v= numel(c);
one_h_output=zeros(10,m);
one_h_dtest=zeros(10,m);
dtest=(dtest)';
output= output';
for p=1:m
    one_h_output(output(p,1),p)=1;
    one_h_dtest(dtest(p,1),p)=1;
end
plotconfusion(one_h_dtest, one_h_output)
%% plot ROC
m= size(datatest,2);
c= unique(dtest);
v= numel(c);
one_h_output=zeros(10,m);
one_h_dtest=zeros(10,m);
dtest=dtest';
output= output';
for p=1:m
    one_h_output(output(1,p),p)=1;
    one_h_dtest(dtest(1,p),p)=1;
end
plotroc(one_h_dtest, one_h_output)