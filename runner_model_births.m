%generate a classifier for predicting the type of labor.
%uses labor_train_test_data.mat
%set 'classifier_type' to 'nb' to get a naive bayes classifier, 'dt' for decision tree, 'knn' for k nearest neighbor.

fprintf('running with classifier type %s\n', classifier_type)
test_subsample = in_test;
if (strcmp(classifier_type,'nb'))
	test_predicted = predict(fitcnb(patterns(in_train, keep_vars), targets(in_train)), patterns(in_test, keep_vars));
else 
    if (strcmp(classifier_type,'dt'))
        test_predicted = predict(fitctree(patterns(in_train, keep_vars), targets(in_train)), patterns(in_test, keep_vars));
    else 
        if (strcmp(classifier_type,'knn'))
            classifier = fitcknn(patterns(in_train, keep_vars), targets(in_train), 'CategoricalPredictors', 'all', 'NumNeighbors', 9, 'Distance', 'seuclidean');
            test_predicted = nan(length(in_test),1);
            test_subsample = 1:length(in_test);
            for j=1:ceil(length(in_test)/10000)
                fprintf('now at round %d out of %d\n', j, ceil(length(in_test)/10000));
                current_idx = ((j-1)*10000+1):min(j*10000,length(in_test));
                tic
                test_predicted(current_idx) = predict(classifier, patterns(in_test(current_idx), keep_vars));
                toc
            end
        else
            print 'unrecognized type'
            return
        end
    end
end

all_labels = unique([test_actual;test_predicted]);
%all_labels are artificially added to verify a full crosstab
ct = crosstab([test_actual(test_subsample);all_labels], [test_predicted;all_labels]);
if (size(ct,1) ~= size(ct,2))
    fprintf('something wrong with crosstab!\n');
end
ct = ct - eye(size(ct,1));
fprintf('Model error: %f\n', 1-sum(diag(ct))/sum(ct(:)));

save labor_classifier test_* in_train in_test utargets


