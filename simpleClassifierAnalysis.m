clear res
res.wg = nan(numProtValues,1);
res.pg = nan(numProtValues,1);
res.pi_g = nan(numProtValues,1);
res.pg_fact = nan(numProtValues,1);
res.pi_g_fact = nan(numProtValues,1);
res.alpha0g = nan(1,numProtValues);
res.alpha1g = nan(1,numProtValues);
for v = 1:numProtValues
    vindices = protectedValuesTest==v;
    res.wg(v) = sum(vindices)/length(vindices);
    res.pi_g(v) = mean(Ytest(vindices)+1)/2;
    res.pg(v) = mean((Ypred(vindices)+1)/2);
    res.pi_g_fact(v) = sum(Ytest(vindices)==1)*divisionfactor/sum(vindices);
    res.pg_fact(v) = sum(Ypred(vindices)==1)*divisionfactor/sum(vindices);
    res.alpha0g(v) = sum((Ytest(vindices) == -1) & (Ypred(vindices) == 1))/sum(Ytest(vindices) == -1);
    res.alpha1g(v) = sum((Ytest(vindices) == 1) & (Ypred(vindices) == -1))/sum(Ytest(vindices) == 1);
end
testaccuracy = mean(Ypred == Ytest);
fprintf('test error = $%.2f\\%%$\n', 100*(1-testaccuracy));


[res.truea0, res.truea1] = find_best_alphas(res.wg, res.pi_g_fact, res.alpha0g, res.alpha1g, divisionfactor);
res.true_unfairness = calcunfairness(res.wg, res.pi_g, res.alpha0g, res.alpha1g, res.truea0, res.truea1);
fprintf('true unfairness = $%.2f\\%%$\n', 100*res.true_unfairness)

i = 1;
clear exp
for beta = [0.01:0.01:1]
    fprintf('beta = %g\n', beta)
    exp(i).beta = beta;
    tolerance = 0.001; %can be reduced, but will take longer to run
    [min_obj, info] = find_lb(res.wg, res.pg_fact, res.pi_g_fact, beta, tolerance, divisionfactor);

    wg = info.wg;
    pg = info.pg;
    p = sum(pg .* wg);
    pi = sum(info.pi_g .* wg);
    pi_g = info.pi_g;
    alpha0g = res.alpha0g; %this is the true one
    alpha1g = res.alpha1g; %this is the true one
    exp(i).besta0g = info.besta0g;
    exp(i).besta1g = info.besta1g;
    alpha0g(isnan(alpha0g)) = 0; %will be multiplied by 0 anyway
    alpha1g(isnan(alpha1g)) = 0; %will be multiplied by 0 anyway
    

    fprintf('minimizing unfairness: $%.2f\\%%$, ratio: %g\n', 100*info.unfairness, res.true_unfairness/info.unfairness);
    fprintf('minimizing error: $%.2f\\%%$\n', 100*info.totalerror)
     %note: the minimizing error will be sum(abs(info.pg-info.pi_g).*wg)
    exp(i).min_obj = min_obj;
    exp(i).unfairness_sol = info.unfairness;
    exp(i).error_sol = info.totalerror;
    exp(i).besta0 = info.besta0;
    exp(i).besta1 = info.besta1;
    i = i+1;
end

