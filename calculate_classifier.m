function all = calculate_classifier(X,categoricalColumns, Y,trainsize, statecol, classifier_type)

%The last parameter is the classifier type. Can be:
% - linear (this is the default)
% - NN (nearest neighbor) 
% - decision tree (tree)

if nargin == 5
    classifier_type = 'linear';
    warning('Using default linear classifier')
end

%Transform categorical variables for those classifiers that can't handle
%categorical variables
if strcmp(classifier_type, 'nn')
    Nvals = ones(size(X, 2), 1);
    for i = 1:size(X, 2)
        if categoricalColumns(i)
            Nvals(i) = length(unique(X(:, i))) - 1;
        end
    end
    
    %Fix categorical columns with more than 2 values
    newX = zeros(size(X, 1), sum(Nvals));
    c    = 0;
    for i = 1:size(X, 2)
        if Nvals(i) == 1
            c = c + 1;
            newX(:, c) = X(:, i);
        else
            [~, ~, curAttribute] = unique(X(:, i));
            curMat  = sparse([1:size(X, 1)]', curAttribute, ones(size(curAttribute)));
            newX(:, c+[1:size(curMat, 2)-1]) = curMat(:, 2:end);
            c       = c + size(curMat, 2)-1;
        end
    end   
    
    X = newX;
end

nsamples    = length(Y);
Xtrain      = X(1:trainsize,:);
Ytrain      = Y(1:trainsize);

Xtest       = X((trainsize+1):end,:);
Ytest       = Y((trainsize+1):end,:);

if strcmp(classifier_type, 'linear')    
    mdl         = fitlm(Xtrain, Ytrain, 'CategoricalVars', categoricalColumns);
elseif strcmp(classifier_type, 'nn') 
    mdl         = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 5);
elseif strcmp(classifier_type, 'tree') 
    mdl         = fitctree(Xtrain, Ytrain, 'CategoricalPredictors', categoricalColumns);
else
    error('Unknown classifier type')
end

Ypredtrain_reals = predict(mdl, Xtrain);

th          = linspace(min(Ypredtrain_reals),max(Ypredtrain_reals),100);
Ypredtrain  = (Ypredtrain_reals > th)*2-1;
trainagreement      = (Ypredtrain == repmat(Ytrain, 1, length(th)));
trainAccuracy_array = mean(trainagreement);

[~, th_ind] = max(trainAccuracy_array);
bestth = th(th_ind);

Ypredtrain = (Ypredtrain_reals > bestth)*2-1; 
Ypred = 2*(predict(mdl, Xtest)> bestth)-1;
all.trainaccuracy = mean(Ypredtrain == Ytrain);
all.testaccuracy = mean(Ypred == Ytest);
all.defaultaccuracy = max(mean(Ytest == 1),1-mean(Ytest==1));
teststate = statecol((trainsize+1):end);
Nteststates = hist(teststate, 1:60);    
divisionfactor = nsamples - trainsize;
for s = 1:60
    sindices{s} = teststate==s; 
    temp.pi_g(s) = mean((Ytest(sindices{s})+1)/2);
    temp.pg(s) = mean((Ypred(sindices{s})+1)/2);
    temp.pi_g_fact(s) = sum(Ytest(sindices{s})==1)*divisionfactor/sum(sindices{s});
    temp.pg_fact(s) = sum(Ypred(sindices{s})==1)*divisionfactor/sum(sindices{s});
    temp.alpha0g(s) = sum((Ytest(sindices{s}) == -1) & (Ypred(sindices{s}) == 1))/sum(Ytest(sindices{s}) == -1);
    temp.alpha1g(s) = sum((Ytest(sindices{s}) == 1) & (Ypred(sindices{s}) == -1))/sum(Ytest(sindices{s}) == 1);
end


ag = Nteststates / (nsamples - trainsize);
sel = Nteststates ~= 0;

all.ag = ag(sel)';
all.pg = temp.pg(sel)';
all.pi_g = temp.pi_g(sel)';
all.pg_fact = temp.pg_fact(sel)';
all.pi_g_fact = temp.pi_g_fact(sel)';
all.divisionfactor = divisionfactor;
all.alpha0g = temp.alpha0g(sel);
all.alpha1g = temp.alpha1g(sel);



