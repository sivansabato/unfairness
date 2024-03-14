%Source: https://www.cdc.gov/nchs/data_access/Vitalstatsonline.htm#Births
n = 4e6;

birth_month = zeros(n, 1);
birth_time  = zeros(n, 1);
birth_dayofweek = zeros(n, 1);
birth_place = cell(n, 1);

mother_age = zeros(n, 1);
mother_nativity = cell(n, 1);
paternity_acknowledged= cell(n, 1);
mother_race = zeros(n, 1);
mother_education = zeros(n, 1);
father_age = zeros(n, 1);
father_race = zeros(n, 1);
father_education = zeros(n, 1);

prior_births_living = zeros(n, 1);
prior_births_dead = zeros(n, 1);
prior_other_terminations = zeros(n, 1);
birth_order = zeros(n, 1);

prenatal_care_bagan = zeros(n, 1);
prenatal_visits = zeros(n, 1);

cigaretts = zeros(n, 4);

mother_height = zeros(n, 1);
mother_bmi = zeros(n, 1);
mother_prepregnancy_weight = zeros(n, 1);
mother_delivery_weight = zeros(n, 1);

prepregnancy_diabetes = cell(n, 1);
gestational_diabetes = cell(n, 1);
prepregnancy_hypertension = cell(n, 1);
gestational_hypertension = cell(n, 1);
prepregnancy_diabeteshypertension_eclampsia = cell(n, 1);
previous_preterm_birth = cell(n, 1);
infertility_treatment = cell(n, 1);
fertility_drugs = cell(n, 1);
previous_cesarean = zeros(n, 1);

gonorrhea = cell(n, 1);
syphilis = cell(n, 1);
chlamydia = cell(n, 1);
gonorrhea = cell(n, 1);
hepb = cell(n, 1);
hepc = cell(n, 1);

labor_induction = cell(n, 1);
labor_augmentation = cell(n, 1);
labor_steroids = cell(n, 1);
labor_antibiotics = cell(n, 1);
labor_chorioamnionitis = cell(n, 1);
labor_anesthesia = cell(n, 1);

delivery_presentation = cell(n, 1);
delivery_method = cell(n, 1);
labor_attendant = cell(n, 1);
labor_payment = cell(n, 1);

baby_sex = cell(n, 1);
apgar_5m = zeros(n, 1);
apgar_10m = zeros(n, 1);
baby_weight = zeros(n, 1);

vars = whos;
vars = {vars.name};
vars = setdiff(vars, {'n'});

c = 0;
fid = fopen('Nat2017PublicUS.c20180516.r20180808.txt');
while 1
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
    
    c = c + 1;
    
    birth_month(c)      = str2double(tline(13:14));
    birth_time(c)       = str2double(tline(19:22));
    birth_dayofweek(c)  = str2double(tline(23));
    
    switch tline(32)
        case '1'
            birth_place{c} = 'Hospital';
        case '2'
            birth_place{c} = 'Birth center';
        case '3'
            birth_place{c} = 'Home (intended)';
        case '4'
            birth_place{c} = 'Home (unintended)';
        case '5'
            birth_place{c} = 'Home (unknown)';
        case '6'
            birth_place{c} = 'Clinic';
        otherwise
            birth_place{c} = 'Other\Unknown';
    end
    
    mother_age(c)       = str2double(tline(75:76));
    
    switch tline(84)
        case '1'
            mother_nativity{c} = 'US';
        case '2'
            mother_nativity{c} = 'non-US';
        otherwise
            mother_nativity{c} = 'Unknown';
    end
    
    mother_race(c)       = str2double(tline(108:109));
    
    switch tline(119)
        case '1'
            paternity_acknowledged{c} = 'Yes';
        case '2'
            paternity_acknowledged{c} = 'No';
        otherwise
            paternity_acknowledged{c} = 'Unknown\NA';
    end
    
    mother_education(c)       = str2double(tline(124));
    
    father_age(c)       = str2double(tline(149:150));
    father_race(c)       = str2double(tline(154:155));
    father_education(c)       = str2double(tline(163));
    
    prior_births_living(c)       = str2double(tline(171:172));
    prior_births_dead(c)       = str2double(tline(173:174));
    prior_other_terminations(c)       = str2double(tline(175:176));
    birth_order(c)       = str2double(tline(182));
    
    prenatal_care_bagan(c)       = str2double(tline(224:225));
    prenatal_visits(c)       = str2double(tline(238:239));
    
    cigaretts(c, 1) = str2double(tline(253:254));
    cigaretts(c, 2) = str2double(tline(255:256));
    cigaretts(c, 3) = str2double(tline(257:258));
    cigaretts(c, 4) = str2double(tline(259:260));
    
    mother_height(c) = str2double(tline(280:281));
    mother_bmi(c) = str2double(tline(283:286));
    mother_prepregnancy_weight(c) = str2double(tline(292:294));
    mother_delivery_weight(c) = str2double(tline(299:301));
    
    prepregnancy_diabetes{c} = tline(313);
    gestational_diabetes{c} = tline(314);
    prepregnancy_hypertension{c} = tline(315);
    gestational_hypertension{c} = tline(316);
    prepregnancy_diabeteshypertension_eclampsia{c} = tline(317);
    previous_preterm_birth{c} = tline(318);
    infertility_treatment{c} = tline(325);
    fertility_drugs{c} = tline(326);
    previous_cesarean(c) = str2double(tline(332:333));
   
    gonorrhea{c} = tline(343);
    syphilis{c} = tline(344);
    chlamydia{c} = tline(345);
    gonorrhea{c} = tline(346);
    hepb{c} = tline(347);
    hepc{c} = tline(348);
   
    labor_induction{c} = tline(383);
    labor_augmentation{c} = tline(384);
    labor_steroids{c} = tline(385);
    labor_antibiotics{c} = tline(386);
    labor_chorioamnionitis{c} = tline(387);
    labor_anesthesia{c} = tline(388);
    
    switch tline(401)
        case '1'
            delivery_presentation{c} = 'Cephalic';
        case '2'
            delivery_presentation{c} = 'Breech';
        otherwise
            delivery_presentation{c} = 'Other\Unknown';
    end
    
    switch tline(402)
        case '1'
            delivery_method{c} = 'Spontaneous';
        case '2'
            delivery_method{c} = 'Forceps';
        case '3'
            delivery_method{c} = 'Vacuum';
        case '4'
            delivery_method{c} = 'Cesarean';
        otherwise
            delivery_method{c} = 'Other\Unknown';
    end
    
    switch tline(433)
        case '1'
            labor_attendant{c} = 'MD';
        case '2'
            labor_attendant{c} = 'DO';
        case '3'
            labor_attendant{c} = 'CNM';
        case '4'
            labor_attendant{c} = 'Other midwife';
        otherwise
            labor_attendant{c} = 'Unknown or other';
    end
    
     switch tline(435)
        case '1'
            labor_payment{c} = 'Medicaid';
        case '2'
            labor_payment{c} = 'Private insurance';
        case '3'
            labor_payment{c} = 'Self-pay';
        case '4'
            labor_payment{c} = 'Indian Health Service';
        case '5'
            labor_payment{c} = 'CHAMPUS\TRICARE';
        case '6'
            labor_payment{c} = 'Other gov';
         otherwise
            labor_payment{c} = 'Other\Unknown';
     end
     
     apgar_5m(c) = str2double(tline(444:445));
     apgar_10m(c) = str2double(tline(448:449));
     
     baby_sex{c} = tline(475);
     baby_weight(c) = str2double(tline(504:507));
     
     if rem(c, 1e4) == 0
         disp(c)
     end
end
fclose(fid);

for i = 1:length(vars)
    eval([vars{i} '=' vars{i} '(1:c, :);']);
end
