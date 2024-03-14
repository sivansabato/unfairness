function all = calculate_classifier_multiclass(X,categoricalColumns, Y,trainsize, statecol, classifier_type)

%The last parameter is the classifier type. Can be:
% - linear (this is the default)
% - NN (nearest neighbor) 
% - decision tree (tree)

if nargin == 5
    classifier_type = 'tree';
    warning('Using default tree classifier')
end

fprintf('DECIMATING for faster run time\n')
trainsize = floor(trainsize/30);
fprintf('train size = %d\n', trainsize)
X = X(1:(2*trainsize), :);
Y = Y(1:(2*trainsize), :);
statecol = statecol(1:(2*trainsize));


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
            curMat  = sparse((1:size(X, 1))', curAttribute, ones(size(curAttribute)));
            newX(:, c+(1:size(curMat, 2)-1)) = curMat(:, 2:end);
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

if strcmp(classifier_type, 'nn') 
    mdl         = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 1);
elseif strcmp(classifier_type, 'tree') 
     mdl         = fitctree(Xtrain, Ytrain, 'CategoricalPredictors', categoricalColumns);
else
    error('Unknown classifier type')
end

Ypredtrain = predict(mdl, Xtrain);
Ypred = predict(mdl, Xtest);
all.trainaccuracy = mean(Ypredtrain == Ytrain);
all.testaccuracy = mean(Ypred == Ytest);
teststate = statecol((trainsize+1):end);
Nteststates = hist(teststate, 1:60); 
sel = Nteststates ~= 0;
existing_states = find(sel);
label_values = 1:max(Ypred); 
for s = existing_states
    %%%%% convert to multiclass
    sindices{s} = teststate==s; 
    temp.pi_g(s,:) = histc(Ytest(sindices{s}), label_values)/sum(sindices{s});
    temp.pg(s,:) = histc(Ypred(sindices{s}), label_values)/sum(sindices{s});
    conmat = confusionmat(Ypred(sindices{s}),Ytest(sindices{s}), 'ORDER', label_values); %this is unnormalized
    %normalize so that the sum over z (the columns) is one
    sum_y = sum(conmat,2);
    zeros_in_y = (sum_y ==0);
    temp.alphag(s,:,:) = conmat./repmat(sum_y,[1,length(label_values)]);
    temp.alphag(s,zeros_in_y,:) = 1/length(label_values);
end


ag = Nteststates / (nsamples - trainsize);

all.ag = ag(sel);
all.pg = temp.pg(sel,:);
all.pi_g = temp.pi_g(sel,:,:);
all.alphag = temp.alphag(sel,:,:);



