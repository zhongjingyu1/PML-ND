function [pre_label, modProb,res_once] = PML_ND_predict(weights, test_feature, test_target)
test_target=test_target';
modProb = test_feature * weights;
pre_label = modProb;
pre_label(pre_label>0.5) = 1;
pre_label(pre_label<=0.5) = 0;
HammingLoss = Hamming_loss(pre_label', test_target');
RankingLoss = Ranking_loss(modProb', test_target');
OneError = One_error(modProb', test_target');
Coverage = coverage(modProb', test_target');
AvgPrecision = Average_precision(modProb', test_target');
MacroF = MacroF1(pre_label',test_target');
MicroF = MicroF1(pre_label',test_target');
res_once = [HammingLoss, RankingLoss, OneError, AvgPrecision];
end