import os
import pytrec_eval
from src import data
from statistics import mean
from scipy.stats import ttest_rel as paired_ttest


def validate(run_scores, valid_qrels, metric="map"):
    validation_metric = metric
    if metric.startswith("P_"):
        metric = "P"
    if metric.startswith("ndcg_cut"):
        metric = "ndcg_cut"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
    eval_scores = trec_eval.evaluate(run_scores)
    return [d[validation_metric] for d in eval_scores.values()]


def main_robust():
    # folds = "F1 F2 F3 F4 F5".split()
    folds = "F1".split()
    pool = "data/robust/qrels"
    sir_scores = []
    vb_scores = []
    for f in folds:
        sir_dir = "out/robust_title-desc_vanillabert_ares-large_{}_sense-aware".format(f)
        vb_dir = "out/robust_title-desc_vanillabert-bert-base_noSA_{}".format(f.lower())
        test_file = "{}.test.results".format(f.lower())
        curr_sir = validate(data.read_run_dict(os.path.join(os.path.join(sir_dir, test_file))),
                            data.read_qrels_dict(pool))
        sir_scores.extend(curr_sir)
        curr_vb = validate(data.read_run_dict(os.path.join(vb_dir, test_file)), data.read_qrels_dict(pool))
        vb_scores.extend(curr_vb)
        tstat, pvalue = paired_ttest(curr_sir, curr_vb)
        print(f, tstat, pvalue, mean(curr_sir), mean(curr_vb))
    tstat, pvalue = paired_ttest(sir_scores, vb_scores)
    print("ALL", tstat, pvalue, mean(sir_scores), mean(vb_scores))


def main_clef():
    qrels = "data/clef_data/zeroshot_test/{}_{}_bm25-rm3_test/qrels.tsv"
    sir_model_path = "out/robust-zeroshot_title-desc_vanillambert_ares-mult-313_sense-aware/" \
                     "zeroshot_results/english_only/{}_{}_bm25-rm3_test.results"
    vb_model_path = "out/robust-zeroshot_title-desc_vanillambert/{}_{}_bm25-rm3_test.results"
    # for lang in "fr de it es".split():
    sir_scores = []
    vb_scores = []
    for lang in "it".split():
        # for year in "2000 2001 2002 2003 2004 2005 2006".split():
        for year in "2000".split():
            # for year in "2000 2001 2002 2003 ".split():
            sir_test = sir_model_path.format(lang, year)
            vb_test = vb_model_path.format(lang, year)
            pool = qrels.format(lang, year)
            if os.path.exists(sir_test) and os.path.exists(vb_test) and os.path.exists(pool):
                curr_sir = validate(data.read_run_dict(sir_test), data.read_qrels_dict(pool))
                curr_vb = validate(data.read_run_dict(vb_test), data.read_qrels_dict(pool))
                sir_scores.extend(curr_sir)
                vb_scores.extend(curr_vb)
                tstat, pvalue = paired_ttest(curr_sir, curr_vb)
                print(lang, year, tstat, pvalue, mean(curr_sir), mean(curr_vb))
    tstat, pvalue = paired_ttest(sir_scores, vb_scores)
    print(lang, "ALL", tstat, pvalue, mean(sir_scores), mean(vb_scores))


if __name__ == "__main__":
    main_robust()
    # main_clef()
