% LW-TSK-FC codes
% Paper: A Lightweight TSK Fuzzy Classifier with Quantitative Equivalent 
% Fuzzy Rules via Adaptive Weighting
% Authors: Jin Cao, Ta Zhou, Saikit Lam, Yuanpeng Zhang, Jiang Zhang, 
% Xinyu Fan, Defeng Sun, Jing Cai
% Update date: 20-May-2025 by Jin 

classdef LW_TSK_FC

    properties (GetAccess = public, SetAccess = private)

        % read-only public properties
        data  % samples x features
        label  % [1 2 3 ...]'
        gaussianKernelWide

    end

    properties (GetAccess = private, SetAccess = private)

        zS  % scale for normalization function
        zC  % center for normalization function
        sS
        sC
        fittedCurve
        statisticAverageValue
        statisticStardirdValue
        weightHidden
        weightsOutput

    end


    methods

        % Constructor
        function obj = LW_TSK_FC(data, label, options)
            arguments  % claim for parameter validation
                data = [];
                label = [];
                options.gaussianKernelWide = 50;
            end

            obj.data = data;
            obj.label = label;
            obj.gaussianKernelWide = options.gaussianKernelWide;

        end

        function [labelPredictedTrain, scorePredictedTrain, obj] = LW_TSK_FC_train(obj)

            % data preprocessing
            % normalization
            [dataTrain, obj.zC, obj.zS] = normalize(obj.data,"zscore");  % zscore or range
            [dataTrain, obj.sC, obj.sS] = normalize(dataTrain, "range");

            [N, d] = size(dataTrain);
            labelTrain = obj.label;
        
            % label transfer
            if min(unique(labelTrain)) ~= 1
                error('Please begin the label with 1.')
            end

            labelTrainMapped = zeros(N, length(unique(labelTrain)));
            for in = 1:N
                % label value is mapped to label vector which are consist of 0 and 1
                labelTrainMapped(in, labelTrain(in)) = 1;
            end

            % antecedent network
            % membership layear
            M = 2;
            U = nan(M, d, N);  % store the membership values of each sample
            for in = 1:N
                U(:, :, in) = [dataTrain(in, :); 1-dataTrain(in, :)];  % the membership value is determined according to the two specific trigular functions.
            end

            % computing the weights of fuzzy rules
            [numRule, obj.fittedCurve] = RankToNum(U, []);  % output = N x (d*M)
            obj.statisticAverageValue = mean(U, 3);
            obj.statisticStardirdValue = std(U, [], 3);
            weightsRule = nan(size(U));
            for in = 1:N
                Ui = (squeeze(U(:, :, in)) - obj.statisticAverageValue) ./ obj.statisticStardirdValue;
                P = normcdf(Ui) .^ numRule(:, :, in);
                P = ((P - min(P, [], "all")) * 0.157) / (max(P, [], "all") - min(P, [], "all")) + 0.01;  % 0.157 and 0.01 are determined according to value range of F(.) function to avoid the 0.5 and too small value for G(.)
                weightsRule(:, :, in) = Ui ./ norminv(P);
            end

            % computing the weighted fuzzy rules
            fuzzyRuleValueWeighted = weightsRule .* U;
            fuzzyRuleValueWeighted_resized = nan(N, d*M);
            for i = 1:N
                 temp = fuzzyRuleValueWeighted(:, :, i);
                 fuzzyRuleValueWeighted_resized(i, :) = temp(:);
            end

            % kernel function
            gaussianvalue = gaussianKernel(fuzzyRuleValueWeighted_resized, obj.gaussianKernelWide);
            gaussianvalue = [gaussianvalue, gaussianvalue(:, 1)];  % N x N+1
            H = nan(N, d+1);  % N x d+1

            for i = 1:N
                % gaussian kernel function
                H(i, :) = [1 dataTrain(i, :)] * gaussianvalue(i, i+1);  % Eq.(34)
            end

            % compute the weights by the theory of LLM
            warning("off")
            obj.weightsOutput = H' * pinv((H*H' + 1/(2*2)*eye(N))) * labelTrainMapped;  % d+1 x R  ((1 ./ t) * I + H' * H) \ H' * T';
            obj.weightsOutput = ((1 ./ 400) * eye(d+1) + H' * H) \ H' * labelTrainMapped; 
            warning("on")

            % compute the output
            scorePredictedTrain = (H * obj.weightsOutput);
            [~, labelPredictedTrain] = max(scorePredictedTrain, [], 2);

        end

        function [labelPredictedTest, scorePredictedTest] = LW_TSK_FC_test(obj, dataTest)

            % normlization
            dataTest = normalize(dataTest, 'center', obj.zC,'scale', obj.zS);
            dataTest = normalize(dataTest, 'center', obj.sC,'scale', obj.sS);
            [N, d] = size(dataTest);

            % fuzzification
            U = nan(2, d, N);  % store the membership values of each sample
            for i = 1:N
                U(:, :, i) = [dataTest(i, :); 1-dataTest(i, :)];
            end

            % Equivalent mapping of fuzzy rules
            M = 2;
            [numRule, ~] = RankToNum(U, obj.fittedCurve);  % 03-Mar-2025 | output = N x (d*M)
            weightsRule = nan(size(U));
            for in = 1:N
                Ui = (squeeze(U(:, :, in)) - obj.statisticAverageValue) ./ obj.statisticStardirdValue;
                P = normcdf(Ui) .^ reshape(numRule(:, :, in), [M d]);
                P = ((P - min(P, [], "all")) * 0.157) / (max(P, [], "all") - min(P, [], "all")) + 0.01;
                weightsRule(:, :, in) = Ui ./ norminv(P);
            end

            fuzzyRuleValueWeighted = weightsRule .* U;
            fuzzyRuleValueWeighted_resized = nan(N, d*M);
            for i = 1:N
                 temp = fuzzyRuleValueWeighted(:, :, i);
                 fuzzyRuleValueWeighted_resized(i, :) = temp(:);
            end

            % kernel function
            gaussianvalue = gaussianKernel(fuzzyRuleValueWeighted_resized, obj.gaussianKernelWide);
            gaussianvalue = [gaussianvalue, gaussianvalue(:, 1)];
            H = nan(N, d+1);  % N x d+1

            for i = 1:N
                % gaussian kernel function
                H(i, :) = [1 dataTest(i, :)] * gaussianvalue(i, i+1);  % Eq.(34)
            end

            % compute the output
            scorePredictedTest = H * obj.weightsOutput;
            [~, labelPredictedTest] = max(scorePredictedTest, [], 2);
        end
    end
end


%% additional functions

function [repeatNum, fittedCurve] = RankToNum(membershipValues, fittedCurve)
%CREATEFIT1(RANK,KEY)
%  the mapping from rank to repeat number.
%
%  INPUT: 
%      membershipValues: M*d*N dimension data
%  OUTPUT:
%      repeatNum: the fitted repetation numbers of membership values
%

[M, d, N] = size(membershipValues);

warning("off")
if isa(fittedCurve, 'fittype')

    repeatNum = nan(size(membershipValues));

    for in = 1:N

        membershipValuesPatient = membershipValues(:, :, in)';
        membershipValuesPatient = membershipValuesPatient(:);

        % The calculation of sorting rate
        [~, numRateIndex] = sort(membershipValuesPatient);
        [~, rankIndex] = ismember(1:d*M, numRateIndex);

        % the repeation numbers
        repeatNum(:, :, in) = reshape(ceil(fittedCurve(rankIndex)), [d M])';

    end

else

    % four specific points used to generate the fitting function
    ranks = [1 2 3 M^d];
    repeatations = [M^(d-1), M^(d-1)/2, M^(d-1)/4, 0];

    [xData, yData] = prepareCurveData(ranks', repeatations);

    % settings
    ft = fittype('a*b^x');
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.StartPoint = [max(xData) 0.5];  % 10/03/2025

    % fitting
    [fittedCurve, ~] = fit( xData, yData, ft, opts );

    repeatNum = nan(size(membershipValues));

    for in = 1:N

        membershipValuesPatient = membershipValues(:, :, in)';
        membershipValuesPatient = membershipValuesPatient(:);

        % The calculation of sorting rate
        [~, numRateIndex] = sort(membershipValuesPatient);
        [~, rankIndex] = ismember(1:d*M, numRateIndex);

        % the repeation numbers
        repeatNum(:, :, in) = reshape(ceil(fittedCurve(rankIndex)), [d M])';

    end

end

warning("on")

end


% Gaussian kernel function
function K = gaussianKernel(X, sigma)

    distances = pdist2(X, X, 'euclidean').^2;    
    K = exp(-distances ./ (2 * repmat(sigma, [1 length(sigma)]) .^ 2));

end
