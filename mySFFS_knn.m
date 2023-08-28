function [sel,performance] = mySFFS_knn(datatrain,dtrain,datatest,dtest)
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
        %% step2: train classifier using data train & train label
        mdl= fitcknn(datatrainv_cond',dtrain,'NumNeighbors',1)
        
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
    plot(bestperformance(1:iter),'b','linewidth',2)
    hold on
    plot(iter,bestperformance(iter),'ro','linewidth',2)
    grid on
    grid minor
    %     text(iter-0.2,bestperformance(iter)+rand(1)*0.3+0.5,num2str(sel),'fontsize',10)
    drawnow
    xlabel('iteration');
    ylabel('performance');
    title('SFFS for knn classifier')
    
end
performance=sort(bestperformance,'descend');
end

