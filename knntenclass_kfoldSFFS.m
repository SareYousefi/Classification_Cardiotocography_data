clc
clear
close all
% groups: 10
data =xlsread ('dataclass');
label=data(:,end);
Totaldata=data(:,1:21);
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
    [sel,performance] = mySFFS_knnten(datatrain,dtrain,datatest,dtest)
    % selection
    numf=1;
    datatrain= datatrain(sel(1:numf),:);
    datatest= datatest(sel(1:numf),:);
    %% step2: train classifier using datatrain
%     k=10;
%     Distance='Cityblock';
%     [MDL] = myknntrain(datatrain,dtrain,k,Distance);
%     [output] = myknnclassify(MDL,datatest);
    mdl= fitcknn(datatrain',dtrain,'NumNeighbors',9,'Distance','chebychev','Standardize',1);
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
    
    
end
disp(['Total Accuracy: ',num2str(mean(accuracy)) ,'%'])
disp(['Accuracy1: ',num2str(mean(accuracy1)) ,'%'])
disp(['accuracy2: ',num2str(mean(accuracy2)) ,'%'])
disp(['accuracy3: ',num2str(mean(accuracy3)) ,'%'])
disp(['accuracy4: ',num2str(mean(accuracy4)) ,'%'])
disp(['accuracy5: ',num2str(mean(accuracy5)) ,'%'])
disp(['accuracy6: ',num2str(mean(accuracy6)) ,'%'])
disp(['accuracy7: ',num2str(mean(accuracy7)) ,'%'])
disp(['accuracy8: ',num2str(mean(accuracy8)) ,'%'])
disp(['accuracy9: ',num2str(mean(accuracy9)) ,'%'])
disp(['accuracy10: ',num2str(mean(accuracy10)) ,'%'])
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