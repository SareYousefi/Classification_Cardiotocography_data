function [MDL] = myknntrain(datatrain,dtrain,k,Distance)
%UNTITLED2 Summary of this function goes here
%Distance= Euclidean ,Cityblock,Chebychev,Minkowski,Cosine,Correlation
MDL.datatrain=datatrain;
MDL.dtrain=dtrain;
MDL.NumNeighbors=k;
MDL.Distance=Distance;
Y= unique(dtrain);
MDL.Y= Y;
end

