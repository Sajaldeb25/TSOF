%% This code is the Tree based Self-Organising Fuzzy (TSOF) classifier
%% Author: Sajal Debnath


clear all
clc
close all

load GM_001.mat

k = randperm(no_of_instance);
Data_Train = GM_data(k(1:for_offine_training),:);  
Label_Train = GM_label(k(1:for_offine_training),:); 


%% The TSOF classifier conducts offline learning from static data
Input.TrainingData=Data_Train;
Input.TrainingLabel=Label_Train;
GranLevel=12;
DistanceType='Cosine';
Mode='OfflineTraining';
[Output0]=TSOF(Input,GranLevel,Mode,DistanceType);

%% The TSOF classifier conducts online learning from streaming data
DTra3 = GM_data(k(for_offine_training+1:for_online_training),:);
LTra3 = GM_label(k(for_offine_training+1:for_online_training),:);

Input=Output0;               
Input.TrainingData=DTra3;    
Input.TrainingLabel=LTra3;   
Mode='EvolvingTraining';
[Output1]=TSOF(Input,GranLevel,Mode,DistanceType);

%% The TSOF classifier conducts validation on testing data
Input=Output1;
Input.TestingData=GM_test_data;
Input.TestingLabel=GM_test_label;
Mode='Validation';
[Output2]=TSOF(Input,GranLevel,Mode,DistanceType);


CM = Output2.ConfusionMatrix
Ltemp=size(CM,1);
ClassificatonAccuracy=sum(sum(CM.*eye(Ltemp)))/sum(sum(CM))


% End of code
