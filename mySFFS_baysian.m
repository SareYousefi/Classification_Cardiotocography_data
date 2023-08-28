function [sel,performance] = mySFFS_baysian(datatrain,dtrain,datatest,dtest)
% Sequentional forward feature selection
indx= 1:size(datatrain,1);
sel=[];
for iter=1:size(datatrain,1)
    c=0;
    for i=indx
        c=c+1;
        indx_cond=[sel,i];
        datatrainv_cond= datatrain(indx_cond,:);
        datatestv_cond= datatest(indx_cond,:);
        c1=find(dtrain==1);
        c2=find(dtrain==2);
        c3=find(dtrain==3);
        pw1=size(c1,2)/size(dtrain,2);
        pw2=size(c2,2)/size(dtrain,2);
        pw3=size(c3,2)/size(dtrain,2);
        prior = [pw1 pw2 pw3];
        %% step2: train classifier using data train & train label
        mdl= fitcnb(datatrainv_cond',dtrain,'prior',prior);
        
        %% step3: test trained classifier
        
        output= predict(mdl, datatestv_cond')';
        %% step4: validation
        C= confusionmat(dtest,output);
        % %% total accuracy
        accuracy= sum(diag(C)) / sum(C(:))*100;
        performance(c)= accuracy;
        
    end
    [performance,ind]=sort(performance,'descend');
    bestperformance(iter)= performance(1);
    sel=[sel,indx(ind(1))];
    indx(ind(1))= [];
    performance=[];
    display(['iteration: ',num2str(iter),' performance: ',num2str(bestperformance(iter))])
    %%ploting
%     plot(bestperformance(1:iter),'b','linewidth',2)
%     hold on
%     plot(iter,bestperformance(iter),'ro','linewidth',2)
%     grid on
%     grid minor
%     %     text(iter-0.2,bestperformance(iter)+rand(1)*0.3+0.5,num2str(sel),'fontsize',10)
%     drawnow
%     xlabel('iteration');
%     ylabel('performance');
%     title('SFFS for Baysian classifier')
    
end
hold on
performance=bestperformance;
end

