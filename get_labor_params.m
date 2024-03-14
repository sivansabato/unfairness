function output = get_labor_params(input_file)
    NUM_CLASSES = 5;
    data = load(input_file);
    gcats = {'test_attendant', 'test_father_race', 'test_mother_race', ...
        'test_payer'};
    
    c = 1;
    for cat = gcats
        myfield = strcat('data.', cat);
        [vals, ~, idx] = unique(eval(myfield{1})); 
        disp(cat)
        disp(vals)
        output(c).numg = length(vals);
        output(c).vals = vals;
       %wg
       wg_count=hist(idx, unique(idx));
       output(c).wg = wg_count/length(idx);
       %pigy
       pigy_count = zeros(output(c).numg, NUM_CLASSES);
       output(c).pigy = pigy_count;
       for y = 1:NUM_CLASSES
           for g = 1:output(c).numg
               pigy_count(g,y) = sum((idx == g) & (data.test_actual == y));
               output(c).pigy(g,y) = pigy_count(g,y)/wg_count(g);
           end
       end
       
       %alphag
       output(c).alphag = zeros(output(c).numg, NUM_CLASSES, NUM_CLASSES);
       for y = 1:NUM_CLASSES
           for z = 1:NUM_CLASSES
               for g = 1:output(c).numg
                    if (pigy_count(g,y)==0)
                        output(c).alphag(g,y,z) = 1/NUM_CLASSES;
                    else
                        output(c).alphag(g,y,z) = sum((idx == g)...
                            & (data.test_actual == y)  & (data.test_predicted == z))/pigy_count(g,y);
                    end
               end
           end
       end
       pg_count = zeros(output(c).numg, NUM_CLASSES);
       output(c).pg = pg_count;
       for y = 1:NUM_CLASSES
           for g = 1:output(c).numg
               pg_count(g,y) = sum((idx == g) & (data.test_predicted == y));
               output(c).pg(g,y) = pg_count(g,y)/wg_count(g);
           end
       end
       
       
       c=c+1;
    end