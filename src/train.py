import os
import torch
import wandb
import random
import argparse
import tempfile
import pickle as pkl
from tqdm import tqdm
from src import data, modeling
import pytrec_eval
from statistics import mean
from collections import defaultdict
# from src.optimizers.radam import RAdam
from src.optimizers.lookahead import Lookahead

SEED = 42
LR = 0.001  # default 0.001
BERT_LR = 2e-5
MAX_EPOCH = 100
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 32  # default 32
GRAD_ACC_SIZE = 1
# other possibilities: ndcg
VALIDATION_METRIC = 'P_20'
# VALIDATION_METRIC = 'map'
PATIENCE = 20  # how many epochs to wait for validation improvement
PACK_DOC_LEN = 800  # Default 800
PACK_QLEN = 100  # Default 20 when no desc (increase to 100 only when using desc)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def model_map(model, bert_name="bert-base-multilingual-cased", args=None, log_retrieval=None):
    if model == 'vanilla_bert':
        return modeling.VanillaBertRanker(bert_name=bert_name)
    elif model == 'sense_vanilla_bert':
        return modeling.VanillaSenseRanker(model_name=bert_name, sense_aware_args=args, log_retrieval=log_retrieval)
    else:
        raise ValueError


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid,
         topic_only=None,
         model_out_dir=None,
         lookahead=False,
         device=None,
         sense_aware=False,
         use_english_only=False):
    """

        Runs the training loop, controlled by the constants above
        Args:
            model(torch.nn.model or str): One of the models in modelling.py,
            or one of the keys of MODEL_MAP.
            dataset: A tuple containing two dictionaries, which contains the
            text of documents and queries in both training and validation sets:
                ({"q1" : "query text 1"}, {"d1" : "doct text 1"} )
            train_pairs: A dictionary containing query document mappings for the training set
            (i.e, document to to generate pairs from). E.g.:
                {"q1: : ["d1", "d2", "d3"]}
            qrels_train(dict): A dictionary containing training qrels. Scores > 0 are considered
            relevant. Missing scores are considered non-relevant. e.g.:
                {"q1" : {"d1" : 2, "d2" : 0}}
            If you want to generate pairs from qrels, you can pass in same object for qrels_train and train_pairs
            valid_run: Query document mappings for validation set, in same format as train_pairs.
            qrels_valid: A dictionary  containing qrels

            #Optional:
            model_out_dir: Location where to write the models. If None, a temporary directory is used.
            topic_only
            lookahead
            device
            sense_aware
            use_english_only
    """

    if isinstance(model, str):
        model = model_map(model).to(device)
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}

    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)
    # optimizer = RAdam([non_bert_params, bert_params])
    if lookahead:
        optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

    epoch = 0
    top_valid_score = None
    top_valid_score_epoch = -1
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train, topic_only, device=device,
                               sense_aware=sense_aware, use_english_only=use_english_only)
        print(f'train epoch={epoch} loss={loss}')
        wandb.log({'epoch': epoch, 'loss': loss})
        if lookahead:
            optimizer._backup_and_load_cache()
            valid_score = validate(model, dataset, valid_run, qrels_valid, epoch, topic_only, device=device,
                                   sense_aware=sense_aware, use_english_only=use_english_only)
            optimizer._clear_and_load_backup()
        else:
            valid_score = validate(model, dataset, valid_run, qrels_valid, epoch, topic_only, device=device,
                                   sense_aware=sense_aware, use_english_only=use_english_only)
        print(f'validation epoch={epoch} score={valid_score}')
        wandb.log({'epoch': epoch, 'valid_score': valid_score})
        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights{}.p'.format(epoch)))
            model.save(os.path.join(model_out_dir, 'best_weights.p'))
            wandb.save(os.path.join(model_out_dir, 'weights{}.p'.format(epoch)))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            model.save(os.path.join(model_out_dir, 'last_weights{}.p'.format(epoch)))
            wandb.save(os.path.join(model_out_dir, 'last_weights{}.p'.format(epoch)))
            break

    # load the final selected model for returning
    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights{}.p'.format(top_valid_score_epoch)), device=device)
    return model, top_valid_score_epoch


def train_iteration(model, optimizer, dataset, train_pairs, qrels, topic_only, device, sense_aware, use_english_only):
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, topic_only, GRAD_ACC_SIZE, PACK_QLEN,
                                            PACK_DOC_LEN, sense_aware):
            batch_languages = record['language'] if 'language' in record else None
            scores = model(record['query_tok'].to(device),
                           record['query_mask'].to(device),
                           record['doc_tok'].to(device),
                           record['doc_mask'].to(device),
                           record['bpe_pos'].to(device),
                           record['retriever_tok'].to(device),
                           record['retriever_tok_mask'].to(device),
                           query_languages=batch_languages,
                           use_english_only=use_english_only)
            count = len(record['query_id']) // 2
            total += count
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])  # pariwise softmax
            loss.backward()
            total_loss += loss.item()
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, valid_qrels, epoch, topic_only, device, sense_aware=False, use_english_only=False):
    run_scores = run_model(model, dataset, run, topic_only,
                           device=device,
                           epoch=epoch,
                           sense_aware=sense_aware,
                           use_english_only=use_english_only)
    metric = VALIDATION_METRIC
    if metric.startswith("P_"):
        metric = "P"
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, {metric})
    eval_scores = trec_eval.evaluate(run_scores)
    # print(eval_scores)
    return mean([d[VALIDATION_METRIC] for d in eval_scores.values()])


# noinspection PyBroadException
def run_model(model, dataset, run, topic_only,
              outpath="",
              desc='valid',
              device=None,
              epoch=-1,
              sense_aware=False,
              use_english_only=False):
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        batch_id = 0
        for records in data.iter_valid_records(model, dataset, run, topic_only,
                                               batch_size=BATCH_SIZE, PACK_QLEN=PACK_QLEN, PACK_DOC_LEN=PACK_DOC_LEN,
                                               sense_aware=sense_aware):
            batch_languages = records['language'] if 'language' in records else None
            scores = model(records['query_tok'].to(device),
                           records['query_mask'].to(device),
                           records['doc_tok'].to(device),
                           records['doc_mask'].to(device),
                           records['bpe_pos'].to(device),
                           records['retriever_tok'].to(device),
                           records['retriever_tok_mask'].to(device),
                           query_languages=batch_languages,
                           batch_id=batch_id,
                           use_english_only=use_english_only)
            batch_id += 1
            if type(scores) == int:
                scores = [scores] * len(records['query_id'])  # Ugly fix for debugging only
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                try:
                    rerank_run[qid][did] = score.item()
                except Exception:  # Ugly fix for debugging only
                    rerank_run[qid][did] = score
            pbar.update(len(records['query_id']))

    if desc == "rerank":
        write_run(rerank_run, outpath)
    try:
        if desc == "valid":
            model.log_retrieval_txt.write("\nCOMPLETED ONE VALIDATION after epoch {}\n".format(epoch))
        if not os.path.exists(model.log_retrieval_cache):
            if model.frozen_query_encoder:
                pkl.dump(model.log_retrieval_dict, open(model.log_retrieval_cache, "wb"))
    except Exception as e:
        print("Model does not support retrieval!", e)
    return rerank_run


def write_run(rerank_run, runf):
    """
        Utility method to write a file to disk. Now unused
    """
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i + 1} {score} run\n')


def main_cli():
    parser = argparse.ArgumentParser('Model training and validation')
    parser.add_argument('--model', choices=['sense_vanilla_bert', 'vanilla_bert'], default='vanilla_bert')
    parser.add_argument('--bert', choices=['bert-base-multilingual-cased', 'bert-base-uncased'],
                        default='bert-base-uncased')
    parser.add_argument('--datafiles', type=str, nargs='+',
                        default=["data/robust/documents.tsv",
                                 "data/robust/title-desc_as_queries.tsv"])
    parser.add_argument('--topics_only', type=str, nargs='+',
                        default=["data/robust/queries.tsv"])
    parser.add_argument('--qrels', type=str, default="data/robust/qrels")
    parser.add_argument('--train_pairs', type=str, default="data/robust/f1.train.pairs")
    parser.add_argument('--valid_run', type=str, default="data/robust/f1.valid.run")
    parser.add_argument('--initial_bert_weights', default=None)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--model_out_dir', default="out/robust_title-desc_vanillabert-bert-base_noSA_f1")
    parser.add_argument('--lookahead_opt', action="store_true")
    parser.add_argument('--sense_aware', action="store_true")
    parser.add_argument('--n_docs', type=int, default=1)
    parser.add_argument('--n_merged', type=int, default=3)
    parser.add_argument('--normalize_query', action="store_true")
    parser.add_argument('--freeze_query_encoder', action="store_true")
    parser.add_argument('--use_english_only', action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)

    wandb.init(project="sir",
               config=args.__dict__)

    wandb.config.update({
        "seed": SEED,
        "lr": LR,
        "bert_lr": BERT_LR,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": BATCHES_PER_EPOCH,
        "grad_acc_size": GRAD_ACC_SIZE,
        "max_epochs": MAX_EPOCH,
        "val_metric": VALIDATION_METRIC,
        "patience": PATIENCE,
        "doc_len": PACK_DOC_LEN,
        "qlen": PACK_QLEN
    })
    if args.sense_aware and not args.model_out_dir.endswith("_sense-aware"):
        args.model_out_dir = args.model_out_dir + "_sense-aware"
    if not os.path.exists(args.model_out_dir):
        os.makedirs(args.model_out_dir)
    if args.sense_aware:
        vc = "-".join(args.valid_run.split("/")[-2:])
        retrieval_cache = os.path.join(args.model_out_dir, vc)
    else:
        retrieval_cache = None
    model = model_map(args.model, bert_name=args.bert, args=args, log_retrieval=retrieval_cache).to(device)
    dataset = data.read_datafiles(args.datafiles)
    topic_only = data.read_datafiles(args.topics_only)[0]
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    print(args.__dict__)
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights, device=device)
    os.makedirs(args.model_out_dir, exist_ok=True)
    wandb.watch(model)
    # we use the same qrels object for both training and validation sets
    main(model, dataset,
         train_pairs,
         qrels,
         valid_run, qrels,
         topic_only,
         args.model_out_dir,
         lookahead=args.lookahead_opt,
         device=device,
         sense_aware=args.sense_aware,
         use_english_only=args.use_english_only)


if __name__ == '__main__':
    main_cli()
