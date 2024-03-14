loaded = 0;
if (exist('data', 'var'))
  %check if already loaded to save time
  if (all(size(data) == [2458285,124]))
    loaded = 1;
  end
end

if (loaded)
   fprintf('data already loaded\n');
else
   fprintf('loading data file...\n');
load('../../DataNotSubmitted//USCensus1990raw.data.mat'); %%%loads data set. It can be downloaded from this link: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)
    fprintf('loaded.\n');
end

if (~exist('classifier_type'))
  classifier_type = 'tree';
fprintf('setting classifier type to default: %s\n', classifier_type);
end
fprintf('classifier type: %s\n', classifier_type);

%%%the split to train and test is random, so results could be different every run.
powstate = 93;
data_screened = data(data(:,powstate) > 0 & data(:,powstate)<90,:);
perm = randperm(size(data_screened,1));
data_screened = data_screened(perm,:);
nsamples = size(data_screened,1);
trainsize = floor(nsamples/2); 


clear catCol
clear output
clear label

for i=1:size(data_screened,2)
    vals{i} = unique(data_screened(:,i));
    catCol(i) = length(vals{i}) < 10;
end



clear output
loc = 0;
for i = 1:length(vals)
    if ((length(vals{i}) > 2) && (length(vals{i}) <= 10))
         fprintf('Taking feature %d for multiclass, number of values = %d\n', i, length(vals{i}));
    else
        continue;
    end
    %convert labels to numbers 1:number_of_labels
    [~,~,Y] = unique(data_screened(:,i));
    h = histc(Y, unique(Y))/length(Y);
    %remove cases that are too unbalanced 
    if (max(h) > 0.95)
        fprintf('skipping due to imbalance\n')
        continue;
    end
     
    X = data_screened;
    X(:,[i,powstate]) = []; %remove these columns
    catColTemp = catCol;
    catColTemp([i,powstate]) = [];
    alltemp = calculate_classifier_multiclass(X,catColTemp, Y, trainsize, data_screened(:,powstate),classifier_type);
    %remove cases that have close to zero error    
    if (alltemp.testaccuracy == 1)
        fprintf('skipping due to triviality\n')
        continue;
    end
    
    
    
    
    loc = loc+1;
    fprintf('========== recording value %d\n', i)
    output(loc).numy = length(vals{i}); %%%% In current saved mat files it is erroneously called numg
    output(loc).wg = alltemp.ag;
    output(loc).pg = alltemp.pg;
    output(loc).pigy = alltemp.pi_g;
    output(loc).trainaccuracy = alltemp.trainaccuracy;
    output(loc).testaccuracy = alltemp.testaccuracy;
    output(loc).alphag = alltemp.alphag;
end


save(['census_multiclass_probs_' classifier_type '.mat'], 'output')


 
