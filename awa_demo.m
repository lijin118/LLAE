%% %%% AwA DEMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% Low-rank Linear Autoencoder %%%%%%%%%%%%%%%%%%%%%
%%%Revised from Semantic Autoencoder for Zero-shot Learning%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc, clear all,  close all


%%%%% Load the data
load('awa_demo_data.mat');
X_tr    = NormalizeFea(X_tr')';


%%%%% Training
lambda  = 6e5;
beta    =8e11;
rank    =36; 


X=X_tr';
S=S_tr';

X_c=corrupt(X',5)'; %data corruption
X_c=NormalizeFea(X_c);

A = S*S';
B = lambda * X_c*X_c';
C = S*X'+lambda*S*X_c';
[k,~]=size(S);
U=0;

for i=1:5
    
    fprintf('iteration: %d \n', i);
    
    W = sylvester(A,B,C);
    [V,~,~] = svd(W*W');
    U=eye(k,k)-V(:,1:rank)*V(:,1:rank)';
    A = S*S' + beta*U;
    
    %%%%% Test %%%%%
    param.HITK           = 1;%1
    param.testclasses_id = param.testclasses_id;
    param.test_labels    = param.test_labels;
    
    %[F --> S], projecting data from feature space to semantic sapce
    S_est        = NormalizeFea(X_te) * NormalizeFea(W)';
    [zsl_accuracy, Y_hit5] = zsl_el((S_est), S_te_gt, param);
    
    fprintf('[1] AwA ZSL accuracy [V >>> S]: %.2f%%\n', zsl_accuracy*100);
    
    %[S --> F], projecting from semantic to visual space
    X_te_pro     = NormalizeFea( S_te_pro')' * NormalizeFea(W);
    [zsl_accuracy]= zsl_el(X_te, X_te_pro, param);
    fprintf('[2] AwA ZSL accuracy [S >>> V]: %.2f%%\n', zsl_accuracy*100);
end













