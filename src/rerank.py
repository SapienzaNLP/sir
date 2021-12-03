import os
import torch
import argparse
from src import train
from src import data
import pytrec_eval
from statistics import mean

from src.data_creation.data_utils import postprocess_output


def validate(run_scores, valid_qrels, metric):
    validation_metric = metric
    if metric.startswith("P_"):
        metric = "P"
    if metric.startswith("ndcg_cut"):
        metric = "ndcg_cut"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
    eval_scores = trec_eval.evaluate(run_scores)
    return mean([d[validation_metric] for d in eval_scores.values()])


def main_cli():
    parser = argparse.ArgumentParser('Model re-ranking')
    parser.add_argument('--bert', choices=['bert-base-multilingual-cased', 'bert-base-uncased'],
                        default='bert-base-uncased')
    parser.add_argument('--model',
                        choices=['vanilla_bert', 'cedr_pacrr', 'cedr_knrm', 'cedr_drmm', 'sense_vanilla_bert'],
                        default='vanilla_bert')
    parser.add_argument('--datafiles', type=str, nargs='+',
                        default=["data/robust/documents.tsv",
                                 "data/robust/title-desc_as_queries.tsv"])
    parser.add_argument('--topics_only', type=str, nargs='+', default=["data/robust/queries.tsv"])
    parser.add_argument('--run', default="data/robust/f1.test.run")
    parser.add_argument('--qrels', type=str, default="data/robust/qrels")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--model_weights', required=True, default="out/robust_title-desc_vanillabert-bert-base_noSA_f1/"
                                                                  "weights10.p")
    parser.add_argument('--out_path', default="f1.test.results")
    parser.add_argument('--n_docs', type=int, default=1)
    parser.add_argument('--n_merged', type=int, default=3)
    parser.add_argument('--normalize_query', action="store_true")
    parser.add_argument('--freeze_query_encoder', action="store_true")
    parser.add_argument('--sense_aware', action="store_true")
    parser.add_argument('--use_english_only', action="store_true")

    args = parser.parse_args()
    args.out_path = os.path.join("/".join(args.model_weights.split("/")[:-1]), args.out_path)
    print(args.out_path)
    device = torch.device(args.device)

    if args.freeze_query_encoder:
        vc = "-".join(args.run.split("/")[-2:])
        retrieval_cache = os.path.join("/".join(args.out_path.split("/")[:-1]), vc)
    else:
        retrieval_cache = None

    model = train.model_map(args.model, bert_name=args.bert, args=args, log_retrieval=retrieval_cache).to(device)
    dataset = data.read_datafiles(args.datafiles)
    topic_only = data.read_datafiles(args.topics_only)[0]
    run = data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load(args.model_weights, device=device)
    train.run_model(model, dataset, run, topic_only, args.out_path,
                    desc='rerank',
                    device=device,
                    sense_aware=args.sense_aware,
                    use_english_only=args.use_english_only)

    for m in ["P_20", "ndcg", "map"]:
        score = validate(data.read_run_dict(args.out_path), data.read_qrels_dict(args.qrels), metric=m)
        print(m, score)
    if args.sense_aware:
        postprocess_output(args, retrieval_cache)


if __name__ == '__main__':
    main_cli()
