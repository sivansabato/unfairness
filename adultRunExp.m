load census1994.mat %from Matlab's Statistics and Machine Learning Toolbox.



adultall = [adultdata;adulttest];
trainsize = size(adultdata,1);
numfeatures = size(adultall,2)-1; %last feature is the label
labelcolumn = size(adultall,2);
X = zeros(size(adultall,1), numfeatures);
Y = adultall(:,labelcolumn);

protOptions = [9,10]; %%%%%protected attribute. race=9. sex=10. 
for col = 1:numfeatures
    [vals, ~, numeric] = unique(adultall(:,col));
    X(:,col) = numeric;
    if any(protOptions==col)
        vals
    end
end
[vals, ~, Y] = unique(Y); %translate to numeric values
Y = Y*2-3; %from 1,2 to 1,-1
%catCols = [2,4,6,7,8,9,10,14];



nsamples    = length(Y);
Xtrain      = X(1:trainsize,:);
Ytrain      = Y(1:trainsize);

Xtest       = X((trainsize+1):end,:);
Ytest       = Y((trainsize+1):end,:);

prior{1} = 'empirical';
prior{2} = [0.7, 0.3];

for priorIndex = 1:2
    fprintf('\n\nrunning with prior:')
    prior{priorIndex}
    classifier = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 9, 'Distance', 'seuclidean', 'Prior', prior{priorIndex});
    Ypred = predict(classifier,Xtest);
      
    divisionfactor = 1;
    
    for prot = protOptions
        fprintf('\n\nprotected attribute: %d\n', prot)
        protectedValues = X(:,prot);
        protectedValuesTrain = protectedValues(1:trainsize);
        protectedValuesTest = protectedValues((trainsize+1):end);
    
        numProtValues = max(protectedValues); %assumes values are consecutive
        
        simpleClassifierAnalysis %script
        filename = sprintf('adultResults%d_%d.mat',prot, priorIndex);
        save(filename, 'exp', 'info', 'res')
    
        fprintf('$%.2f\\%%$ & $%.2f\\%%$ & $%.2f\\%%$ \\\\ \n', 100*[info.wg, info.pi_g, info.pg]')
    end
end


