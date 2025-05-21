% LW-TSK-FC application demo
% Paper: A Lightweight TSK Fuzzy Classifier with Quantitative Equivalent
% Fuzzy Rules via Adaptive Weighting
% Authors: Jin Cao, Ta Zhou, Saikit Lam, Yuanpeng Zhang, Jiang Zhang,
% Xinyu Fan, Defeng Sun, Jing Cai
% Update date: 21-May-2025 by Jin

clear
clc


%% Parameters

kfolder = 5;
numRepetitions = 10;
datafilepath = '.\dataset\Vehicle.mat';


%% Start to evaluate

load(datafilepath)  % Load dataset

% Initialize the label
labelID = unique(DataSet(:, 1));
label = DataSet(:, 1) - (min(labelID) - 1) * (min(labelID) ~= 1);

% Randomize the order of samples
randIdx = randperm(size(DataSet, 1));
data = DataSet(randIdx, 2:end);
label = label(randIdx);

for it = 1:numRepetitions  % parfor
    resultsAll = cell(1, 1);
    cvp = cvpartition(label, "KFold", kfolder);

    for iv = 1:cvp.NumTestSets
        trainIdx = cvp.training(iv);
        testIndex = cvp.test(iv);
        dataTrainBalanced = data(trainIdx, :);
        labelTrainBalanced = label(trainIdx);

        % Feature reduction process
        bestAcc = 0;
        featIdx = fscmrmr(dataTrainBalanced, labelTrainBalanced);
        bestFeatIdx = featIdx;
        decreaseCount = 0;
        for numFeat = length(featIdx):-1:1
            currentFeatIdx = featIdx(1:numFeat);
            currentDataTrain = dataTrainBalanced(:, currentFeatIdx);

            % Train the model
            try
                LWTSKmodel = LW_TSK_FC(currentDataTrain, labelTrainBalanced);
                [labelPredictedTrain, ~, ~] = LW_TSK_FC_train(LWTSKmodel);
            catch
                continue
            end

            % Assess the accuracy
            currentAcc = 1 - nnz(labelPredictedTrain ~= labelTrainBalanced) / length(labelPredictedTrain);

            if currentAcc > bestAcc
                bestAcc = currentAcc;
                bestFeatIdx = currentFeatIdx;
                decreaseCount = 0;
            else
                decreaseCount = decreaseCount + 1;
                if decreaseCount >= 2
                    break;
                end
            end
        end

        dataTrainBalanced = dataTrainBalanced(:, bestFeatIdx);
        dataTestBalanced = data(testIndex, bestFeatIdx);
        labelTestBalanced = label(testIndex);

        resultsAll{1}(iv, 1:2) = [it iv];

        % Train the model
        tic
        LWTSKmodel = LW_TSK_FC(dataTrainBalanced, labelTrainBalanced);
        [labelPredictedTrain, scorePredictedTrain, LWTSKmodel] = LW_TSK_FC_train(LWTSKmodel);
        resultsAll{1}(iv, 9) = toc;

        % Test the model
        tic
        [labelPredictedTest, scorePredictedTest] = LW_TSK_FC_test(LWTSKmodel, dataTestBalanced);
        resultsAll{1}(iv, 10) = toc;

        % Assess the accuracy
        resultsAll{1}(iv, 3) = 1 - nnz(labelPredictedTrain ~= labelTrainBalanced) / length(labelPredictedTrain);
        resultsAll{1}(iv, 5) = 1 - nnz(labelPredictedTest ~= labelTestBalanced) / length(labelPredictedTest);

        % Label mapping for AUC assessment
        dataTMappingTrain = full(sparse(1:length(labelTrainBalanced), labelTrainBalanced, 1));
        dataTMappingTest = full(sparse(1:sum(testIndex), labelTestBalanced, 1));

        % training auc value
        aucValues = nan(1, length(labelID));
        for ir = 1:length(labelID)
            [~, ~, ~, aucValues(ir)] = perfcurve(dataTMappingTrain(:, ir), scorePredictedTrain(:, ir), 1);
        end
        resultsAll{1}(iv, 4) = mean(aucValues);

        % testing auc value
        aucValues = nan(1, length(labelID));
        for ir = 1:length(labelID)
            [~, ~, ~, aucValues(ir)] = perfcurve(dataTMappingTest(:, ir), scorePredictedTest(:, ir), 1);
        end
        resultsAll{1}(iv, 6) = mean(aucValues);

        % Specificity and sensitivity for the testing task of binary classification
        if length(labelID) == 2
            C = confusionmat(labelTestBalanced, labelPredictedTest);
            resultsAll{1}(iv, 7) = C(2,2) / (C(2,2) + C(2,1));  % Sensitivity
            resultsAll{1}(iv, 8) = C(1,1) / (C(1,1) + C(1,2));  % Specificity
        end

        fprintf('%s : it %d, iv %d: ACC %.2f, AUC %.2f. \n', datetime('now'), it, iv, resultsAll{1}(iv, 5), resultsAll{1}(iv, 6));

    end

    % Save results
    filePath = fullfile('.\temp\', sprintf('results_%d.mat', it));
    saveResults(filePath, resultsAll);
end

resultsData = [];
for ip = 1:numRepetitions
    load(['.\temp\', sprintf('results_%d.mat', ip)])
    resultsData = [resultsData; results{1,1}];
end

resultsData = [resultsData; mean(resultsData); std(resultsData)];

variableNames = {'Repetition' 'folder' 'TrAcc' 'TrAUC' 'TeACC' 'TeAUC' 'Sen' 'Spe' 'TrTime' 'TeTime'};
results_data = array2table(resultsData, 'VariableNames', variableNames);
writetable(results_data, sprintf('results_%s.csv', datestr(now, 'yyyymmdd_HHMM')));

