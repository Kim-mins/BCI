clc; clear; close all;

grasp_files = dir('..\ConvData\grasp\');
twist_files = dir('..\ConvData\twist\');

files = {};
% 13 subjects in total
% i = 1: ., i = 2: ..
for i = 3:length(grasp_files)
    for j = 3:length(twist_files)
        grasp_file = strsplit(grasp_files(i).name, '_');
        twist_file = strsplit(twist_files(j).name, '_');
        if strcmp(grasp_file(1), twist_file(1)) && strcmp(grasp_file(2), twist_file(2))
            files{end+1} = [char(grasp_file(1)) '_' char(grasp_file(2))];
            continue;
        end 
    end
end
% fse: starting frequency and ending frequency([start, end])
fse = [4 40]; % Hz
% fw: width of each frequency band
fw = [3 4 7 8 11 12 13 15];
% fws: width of each window shift
fws = [2 2 4 5 6 6 5 5];
% order for bandpass filter
ORDER = 3;
% time interval for sampling [from to]
TIME_INTERVAL = [500 2500];
% k for k-fold cross validation
K = 5;
% # of trials
TRIAL_NUM = 200;
% # of channels
CHANNEL_NUM = 64;
% # of classes
CLASS_NUM = 5;
% level of frequency
level_of_frequency = length(fw);
% # of frequency bands
BAND_NUM = 68;
% upper & lower m rows(for feature selection)
M = 2;
% # of pair-wise matrices(CLASS_NUM_P_2)
PW_MAT_NUM = CLASS_NUM*(CLASS_NUM-1);

for i = 1:length(files)
    
    % load data
    [cnt_g, mrk_g, mnt_g] = eegfile_loadMatlab(['..\ConvData\grasp\' files{i} '_grasp_MI.mat']);
    [cnt_t, mrk_t, mnt_t] = eegfile_loadMatlab(['..\ConvData\twist\' files{i} '_twist_MI.mat']);
    % dummy code for extracting TRIAL_NUM
    cnt_g = proc_filtButter(cnt_g, ORDER, fse);
    cnt_t = proc_filtButter(cnt_t, ORDER, fse);
    epo_g = cntToEpo(cnt_g, mrk_g, TIME_INTERVAL); 
    epo_t = cntToEpo(cnt_t, mrk_t, TIME_INTERVAL);
    % define the # of trials
    TRIAL_NUM = size(epo_g.x, 3) + size(epo_t.x, 3);
    % features: result of augmented csp
    % features.*_x:
    % (# of folds, # of frequency bands, # of pair-wise matrix * dimension
    %  of feature vector(for each pair-wise matrix), # of trials)
    % feature.*_y:
    % (# of folds, # of classes, # of trials)
    features.train_x = zeros(K, BAND_NUM, (2*CLASS_NUM-2)*(2*M), TRIAL_NUM*(K-1)/K);
    features.train_y = zeros(K, CLASS_NUM, TRIAL_NUM*(K-1)/K);
    features.test_x = zeros(K, BAND_NUM, (2*CLASS_NUM-2)*(2*M), TRIAL_NUM*1/K);
    features.test_y = zeros(K, CLASS_NUM, TRIAL_NUM*1/K);
    % 5-fold cross validation
    % T: train data, F: test data
    % fold = 1: F T T T T
    % fold = 2: T F T T T
    % fold = 3: T T F T T
    % fold = 4: T T T F T
    % fold = 5: T T T T F
    % But not randomly selected for each fold
    for fold = 1:K
        % count: # of frequency bands
        count = 0;
        % for each level_of_frequency
        for j = 1:level_of_frequency
            % select band width
            % lower-bound
            low = fse(1);
            % upper-bound
            high = low + fw(j);
            % iterate every boundary with fw(i), fws(i)
            while high <= fse(2)
                % band pass filter
                % g for grasp
                cnt_g = proc_filtButter(cnt_g, ORDER, [low high]);
                % t for twist
                cnt_t = proc_filtButter(cnt_t, ORDER, [low high]);
                count = count + 1;
                fprintf('sub %d, fold %d, count %d\n', i, fold, count);
                % update upper, lower-bound
                low = low + fws(j);
                high = low + fw(j);
                % cntToEpo: segmentation of continuous data in 'epochs' based on markers
                epo_g = cntToEpo(cnt_g, mrk_g, TIME_INTERVAL); 
                epo_t = cntToEpo(cnt_t, mrk_t, TIME_INTERVAL);

                % After segmentation, result is epo.*(x, y, t)
                % x: signal data [time interval * features * trials]
                % y: label [classes * trials]
                % t: time [1 * time interval] --> check it yourself
                % total 200 trials --> grasp, twist 100 each

                % twist
                % extract 'Left' class from data
                [epo_left, ev_1] = proc_selectClasses(epo_t, 'Left');
                % extract 'Right' class from data
                [epo_right, ev_2] = proc_selectClasses(epo_t, 'Right');
                % extract 'Rest' class from data
                [epo_rest, ev_3] = proc_selectClasses(epo_t, 'Rest');
                % grasp
                % extract 'Grasp' class from data
                [epo_grasp, ev_4] = proc_selectClasses(epo_g, 'Grasp');
                % extract 'Open' class from data
                [epo_open, ev_5] = proc_selectClasses(epo_g, 'Open');
                % extract 'Rest' class from data
                [epo_rest2, ev_6] = proc_selectClasses(epo_g, 'Rest');

                % siz: size for one fold
                siz = size(epo_left.y, 2) / K;
                % rest is double to others --> 2 times more -->
                % left, right, grasp, open --> 5 each
                % rest, rest2 --> 10 each
                % total 40(dimension[-1]) data(trials)
                test.x = cat(3, epo_left.x(:,:,siz*(fold-1)+1:siz*fold), epo_right.x(:,:,siz*(fold-1)+1:siz*fold), epo_grasp.x(:,:,siz*(fold-1)+1:siz*fold), epo_open.x(:,:,siz*(fold-1)+1:siz*fold), epo_rest.x(:,:,siz*2*(fold-1)+1:siz*2*fold), epo_rest2.x(:,:,siz*2*(fold-1)+1:siz*2*fold));
                test.y = zeros(CLASS_NUM, size(test.x, 3));
                % -1 for last class(more data for last class)
                for k = 1:CLASS_NUM-1
                    test.y(k, siz*(k-1)+1:siz*k) = 1;
                end
                % for last class(rest+rest2)
                test.y(5, siz*4+1:size(test.y, 2)) = 1;
                % As a result, only 5 classes left.
                
                % complie remainders except 'test' data
                train.x = cat(3, epo_left.x(:,:,1:siz*(fold-1)), epo_left.x(:,:,siz*fold+1:size(epo_left.y, 2)), epo_right.x(:,:,1:siz*(fold-1)), epo_right.x(:,:,siz*fold+1:size(epo_right.y, 2)), epo_grasp.x(:,:,1:siz*(fold-1)), epo_grasp.x(:,:,siz*fold+1:size(epo_grasp.y, 2)), epo_open.x(:,:,1:siz*(fold-1)), epo_open.x(:,:,siz*fold+1:size(epo_open.y, 2)), epo_rest.x(:,:,1:siz*2*(fold-1)), epo_rest.x(:,:,siz*2*fold+1:size(epo_rest.y, 2)), epo_rest2.x(:,:,1:siz*2*(fold-1)), epo_rest2.x(:,:,siz*2*fold+1:size(epo_rest2.y, 2)));
                % init train.y(labels)
                train.y = zeros(CLASS_NUM, size(train.x, 3));
                % train.y:
                % 1-dimension: 'k'th-class
                % 2-dimension: 
                %   1~20: left,  21~40: right, 41~60: grasp
                %   61~80: open, 81~120: rest, 121~160: rest2
                for k = 1:CLASS_NUM-1
                    train.y(k,siz*4*(k-1)+1:siz*4*k) = 1;
                end
                % for last class(rest+rest2)
                train.y(5, siz*4*4+1:size(train.x, 3)) = 1;
                % allocate label for trials
                features.train_y(fold,:,:) = train.y(:,:);
                features.test_y(fold,:,:) = test.y(:,:);
                
                % 5-class
                s.train = [0 20 40 60 80 160];
                s.test = [0 5 10 15 20 40];
                % init temporary value
                tmp.train_x = zeros(BAND_NUM, PW_MAT_NUM*(2*M), TRIAL_NUM*(K-1)/K); 
                tmp.test_x = zeros(BAND_NUM, PW_MAT_NUM*(2*M), TRIAL_NUM*1/K);
                cnt = 0;
                % for each pair of classes...(based on pseudocode of paper)
                for k = 1:CLASS_NUM
                    for l = 1:CLASS_NUM
                        if k ~= l
                            % cnt counts the # of pair-wise matrix
                            cnt = cnt+1;
                            % train_pair.x collects train data of class k, l.
                            train_pair.x = cat(3, train.x(:,:,s.train(k)+1:s.train(k+1)), train.x(:,:,s.train(l)+1:s.train(l+1)));
                            % train_pair.y collects labels 
                            % corresponds to train.x.
                            train_pair.y = zeros(2, size(train_pair.x,3));
                            train_pair.y(1,:) = cat(2, ones(1,s.train(k+1)-s.train(k)), zeros(1,s.train(l+1)-s.train(l)));
                            train_pair.y(2,:) = cat(2, zeros(1,s.train(k+1)-s.train(k)), ones(1,s.train(l+1)-s.train(l)));
                            % test_pair.x collects test data of class k, l.
                            test_pair.x = cat(3, test.x(:,:,s.test(k)+1:s.test(k+1)), test.x(:,:,s.test(l)+1:s.test(l+1)));
                            % test_pair.y collects labels
                            % corresponds to test.x.
                            test_pair.y = zeros(2, size(test_pair.x,3));
                            test_pair.y(1,:) = cat(2, ones(1,s.test(k+1)-s.test(k)), zeros(1,s.test(l+1)-s.test(l)));
                            test_pair.y(2,:) = cat(2, zeros(1,s.test(k+1)-s.test(k)), ones(1,s.test(l+1)-s.test(l)));                        
                            % channel labels
                            % 1. channel label이 뭐지?
                            train_pair.clab = [k l];
                            test_pair.clab = [k l];
                            % csp for class pair
                            [pairwise.train, pairwise_w] = proc_csp3(train_pair);
                            % feature selection
                            % --> 2*m rows in total
                            tmp_z = cat(2, pairwise.train.x(:,1:M,:), pairwise.train.x(:,end-M+1:end,:));
                            % 2. 여기서 차원 잘 모름
                            % squeeze flattens dimension with 1
                            % second parameter 1 of var normalizes data
                            Fc_train = squeeze(log(var(tmp_z, 1)));
                            % Fc: (2*m) * trials
                            % collect each pair-wise feature(train)
                            tmp.train_x(count,(cnt-1)*2*M+1:cnt*2*M,s.train(k)+1:s.train(k+1)) = Fc_train(:,1:s.train(k+1)-s.train(k));
                            tmp.train_x(count,(cnt-1)*2*M+1:cnt*2*M,s.train(l)+1:s.train(l+1)) = Fc_train(:,s.train(k+1)-s.train(k)+1:end);
                            % same for test
                            pairwise.test = proc_linearDerivation(test_pair, pairwise_w, 'prependix', 'csp');
                            % based on previous 'M'
                            tmp_z = cat(2, pairwise.test.x(:,1:M,:), pairwise.test.x(:,end-M+1:end,:));
                            % same with train
                            Fc_test = squeeze(log(var(tmp_z, 1)));
                            % collect each pair-wise feature(test)
                            tmp.test_x(count,(cnt-1)*2*M+1:cnt*2*M,s.test(k)+1:s.test(k+1)) = Fc_test(:,1:s.test(k+1)-s.test(k));
                            tmp.test_x(count,(cnt-1)*2*M+1:cnt*2*M,s.test(l)+1:s.test(l+1)) = Fc_test(:,s.test(k+1)-s.test(k)+1:end);
                        end
                    end
                end
                % ez_* is array tmp.*_x with no zeros
                ez_train = nonzeros(tmp.train_x)';
                ez_test = nonzeros(tmp.test_x)';
                features.train_x(fold,count,:,:) = reshape(ez_train, [(2*CLASS_NUM-2)*(2*M), TRIAL_NUM*(K-1)/K]);
                features.test_x(fold,count,:,:) = reshape(ez_test, [(2*CLASS_NUM-2)*(2*M), TRIAL_NUM*1/K]);
            end
        end
    end
    save(['..\RA\5c_f\acsp\' files{i} '.mat'], 'features')
end