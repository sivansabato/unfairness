%generate test/train split

relevant_fields = {'birth_month', 'birth_dayofweek', 'birth_month', 'birth_order', 'birth_place', ...
                   'birth_time', 'chlamydia', 'father_age', 'father_education', ...
                   'fertility_drugs', 'gestational_diabetes', 'gestational_hypertension', 'gonorrhea', 'hepb', ...
                   'hepc', 'infertility_treatment', 'mother_age', 'mother_bmi', 'mother_delivery_weight', ...
                   'mother_education', 'mother_height', 'mother_nativity', 'mother_prepregnancy_weight', ... %'paternity_acknowledged', ...
                   'prenatal_care_bagan', 'prenatal_visits', 'prepregnancy_diabetes', 'prepregnancy_diabeteshypertension_eclampsia', 'prepregnancy_hypertension', ...
                   'previous_cesarean', 'previous_preterm_birth', 'prior_births_dead', 'prior_births_living', 'prior_other_terminations', 'syphilis', 'cigaretts'};
    
%Train\test split
in          = randperm(length(birth_month));
in_train    = in(1:floor(length(in)/2));
in_test     = in(1+floor(length(in)/2):end);

[utargets, ~, targets] = unique(delivery_method);

patterns    = zeros(length(in), length(relevant_fields)-1);
for i = 1:length(relevant_fields)-1
    if iscell(eval(relevant_fields{i}))
        [~, ~, patterns(:, i)] = unique(eval(relevant_fields{i}));
    else
        patterns(:, i) = eval(relevant_fields{i});
    end
end

patterns = [patterns, cigaretts];

keep_vars = find(var(patterns) > 0);

test_actual = targets(in_test);
test_mother_race = mother_race(in_test);
test_father_race = father_race(in_test);
test_payer = labor_payment(in_test);
test_attendant = labor_attendant(in_test);
test_variables = relevant_fields(keep_vars(keep_vars<=length(relevant_fields)));
