import os
import torch
import argparse
from src import train
from src import data
import pytrec_eval
from statistics import mean

from src.data_creation.data_utils import postprocess_output


def validate(run_scores, valid_qrels, metric):
    VALIDATION_METRIC=metric
    if metric.startswith("P_"):
        metric = "P"
    if metric.startswith("ndcg_cut"):
        metric = "ndcg_cut"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
    eval_scores = trec_eval.evaluate(run_scores)
    return mean([d[VALIDATION_METRIC] for d in eval_scores.values()])

def main_cli():
    parser = argparse.ArgumentParser('Model re-ranking')
    parser.add_argument('--bert', choices=['bert-base-multilingual-cased', 'bert-base-uncased'],
                        default='bert-base-multilingual-cased')
    parser.add_argument('--model',
                        choices=['vanilla_bert', 'cedr_pacrr', 'cedr_knrm', 'cedr_drmm', 'sense_vanilla_bert'],
                        default='vanilla_bert')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--model_weights', required=True, default="out/robust-zeroshot_title-desc_vanillambert/weights10.p")
    parser.add_argument('--n_docs', type=int, default=1)
    parser.add_argument('--n_merged', type=int, default=3)
    parser.add_argument('--normalize_query', action="store_true")
    parser.add_argument('--freeze_query_encoder', action="store_true")
    parser.add_argument('--use_english_only', action="store_true")
    parser.add_argument('--sense_aware', action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = os.path.join("/".join(args.model_weights.split("/")[:-1]), "zeroshot_results")
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    all_test_dir = "data/clef_data/zeroshot_test"
    for alg in ["bm25-rm3"]:
        with open(os.path.join(out_dir, "{}_{}_results.txt".format(alg, "title")), "w") as outfile:
            for lang in ["it", "de", "fr", "es"]:
                for year in "2000 2001 2002 2003 2004 2005 2006 2007".split():

                    curr_dir = "{}_{}_{}_test".format(lang, year, alg)
                    run = os.path.join(all_test_dir,curr_dir, "title_test.run")
                    qrels = os.path.join(all_test_dir, curr_dir, "qrels.tsv")
                    if not os.path.exists(run) or not os.path.exists(qrels): continue
                    args.out_path = os.path.join("/".join(args.model_weights.split("/")[:-1]),
                                                 "{}.results".format(curr_dir))
                    print("Working with", lang, year, args.out_path)
                    args.datafiles = [os.path.join(all_test_dir, curr_dir, "title_documents.tsv"), os.path.join(all_test_dir, curr_dir, "title-desc_as_query.tsv") ]
                    args.topics_only = [os.path.join(all_test_dir, curr_dir, "title_as_query.tsv")]
                    args.run=run
                    args.qrels=qrels

                    if args.freeze_query_encoder:
                        vc = "-".join(args.run.split("/")[-2:])
                        retrieval_cache = os.path.join("/".join(args.out_path.split("/")[:-1]), vc)
                    else:
                        retrieval_cache = None

                    model = train.MODEL_MAP(args.model, bert_name=args.bert, args=args, log_retrieval=retrieval_cache).to(device)
                    dataset = data.read_datafiles(args.datafiles)
                    topic_only = data.read_datafiles(args.topics_only)[0]
                    run = data.read_run_dict(args.run)
                    if args.model_weights is not None:
                        model.load(args.model_weights, device=device)
                    train.run_model(model, dataset,
                                    run, topic_only,
                                    args.out_path, desc='rerank',
                                    device=device,
                                    sense_aware=args.sense_aware,
                                    use_english_only=args.use_english_only)
                    scores = []
                    for m in ["P_20", "ndcg", "map"]:
                        s = validate(data.read_run_dict(args.out_path), data.read_qrels_dict(args.qrels), metric=m)
                        scores.append(str(s))
                    outfile.write("{}_{}\t{}\n".format(lang, year, " ".join(scores)))
                    if args.sense_aware:
                        postprocess_output(args, retrieval_cache)

if __name__ == '__main__':
    main_cli()
