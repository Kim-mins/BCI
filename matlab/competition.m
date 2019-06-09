clc; clear; close all;
% 논문에 나온 competition data를 사용해서 acsp를 적용하는 코드입니다
% your path
path = 'D:\BCI-comparative-analysis\BCICIV_2a_gdf\train\mat\';
% files in directory
files = dir(path);
% third file(1: ., 2: .., 3: data.mat)
f_name = files(4).name;
% load data
data = load(strcat(path, f_name));
% channel label
data.clab = cellstr(data.clab)';
% fse: starting frequency and ending frequency([start, end])
fse = [4 40]; % Hz
% fw: width of each frequency band
fw = [3 4 7 8 11 12 13 15];
% fws: width of each window shift
fws = [2 2 4 5 6 6 5 5];
% k for k-fold cross validation
K = 4;
% # of trials
TRIAL_NUM = 288;
% # of channels
CHANNEL_NUM = 22;
% # of classes
CLASS_NUM = 4;
% level of frequency
level_of_frequency = length(fw);
% # of frequency bands
BAND_NUM = 68;
% upper & lower m rows(for feature selection)
M = 4;
% # of pair-wise matrices(CLASS_NUM_P_2)
PW_MAT_NUM = CLASS_NUM*(CLASS_NUM-1);
% # of subjects
SUBJECT_NUM = 9;
% features: result of augmented csp
% features.*_x:
% (# of folds, # of frequency bands, # of pair-wise matrix * dimension
%  of feature vector(for each pair-wise matrix), # of trials)
% feature.*_y:
% (# of folds, # of classes(2, since pair-wise), # of trials)
features.train_x = zeros(SUBJECT_NUM, K, BAND_NUM, (2*CLASS_NUM-2)*(2*M), TRIAL_NUM*(K-1)/K);
features.train_y = zeros(SUBJECT_NUM, K, CLASS_NUM, TRIAL_NUM*(K-1)/K);
features.test_x = zeros(SUBJECT_NUM, K, BAND_NUM, (2*CLASS_NUM-2)*(2*M), TRIAL_NUM*1/K);
features.test_y = zeros(SUBJECT_NUM, K, CLASS_NUM, TRIAL_NUM*1/K);

for sub = 1:SUBJECT_NUM
    for fold = 1:K
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
                count = count + 1;
                fprintf('sub %d, fold %d, count %d\n', sub, fold, count);
                % update upper, lower-bound
                low = low + fws(j);
                high = low + fw(j);
                left = squeeze(data.data(sub,:,:,1:72));
                right = squeeze(data.data(sub,:,:,73:144));
                foot = squeeze(data.data(sub,:,:,145:216));
                tongue = squeeze(data.data(sub,:,:,217:288));
                % siz: size for one fold
                siz = size(left, 3) / K;
                % rest is double to others --> 2 times more -->
                % left, right, grasp, open --> 5 each
                % rest, rest2 --> 10 each
                % total 40(dimension[-1]) data(trials)
                test.x = cat(3, left(:,:,siz*(fold-1)+1:siz*fold), right(:,:,siz*(fold-1)+1:siz*fold), foot(:,:,siz*(fold-1)+1:siz*fold), tongue(:,:,siz*(fold-1)+1:siz*fold));
                test.y = zeros(CLASS_NUM, size(test.x, 3));
                % -1 for last class(more data for last class)
                for k = 1:CLASS_NUM
                    test.y(k, siz*(k-1)+1:siz*k) = 1;
                end
                
                % complie remainders except 'test' data
                train.x = cat(3, left(:,:,1:siz*(fold-1)), left(:,:,siz*fold+1:size(left, 3)), right(:,:,1:siz*(fold-1)), right(:,:,siz*fold+1:size(right, 3)), foot(:,:,1:siz*(fold-1)), foot(:,:,siz*fold+1:size(foot, 3)), tongue(:,:,1:siz*(fold-1)), tongue(:,:,siz*fold+1:size(tongue, 3)));
                % init train.y(labels)
                train.y = zeros(CLASS_NUM, size(train.x, 3));
                % train.y:
                % 1-dimension: 'k'th-class
                % 2-dimension: 
                %   1~20: left,  21~40: right, 41~60: grasp
                %   61~80: open, 81~120: rest, 121~160: rest2
                for k = 1:CLASS_NUM
                    train.y(k,siz*3*(k-1)+1:siz*3*k) = 1;
                end
                % allocate label for trials
                features.train_y(sub,fold,:,:) = train.y(:,:);
                features.test_y(sub,fold,:,:) = test.y(:,:);
                
                % 4-class
                s.train = [0 54 108 162 216];
                s.test = [0 18 36 54 72];
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
                features.train_x(sub,fold,count,:,:) = reshape(ez_train, [(2*CLASS_NUM-2)*(2*M), TRIAL_NUM*(K-1)/K]);
                features.test_x(sub,fold,count,:,:) = reshape(ez_test, [(2*CLASS_NUM-2)*(2*M), TRIAL_NUM*1/K]);
            end
        end
    end
end
save(strcat(path,'comp\features.mat'), 'features');
