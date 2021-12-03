import itertools
import random
from tqdm import tqdm
import torch


def read_datafiles(files):
    queries = {}
    docs = {}
    for file in files:
        for line in tqdm(open(file, "rt"), desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text
            if c_type == 'doc':
                docs[c_id] = c_text
    return queries, docs


def read_qrels_dict(file):
    result = {}
    for line in open(file, "rt"):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(open(file, "rt"), desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(open(file, "rt"), desc='loading pairs (by line)', leave=False):
        qid, doc_id = line.split()
        result.setdefault(qid, {})[doc_id] = 1
    return result


def iter_train_pairs(model, dataset, train_pairs, qrels, topic_only, batch_size, PACK_QLEN, PACK_DOC_LEN,
                     sense_aware=False):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'retriever_tok': [], 'doc_tok': [], 'bpe_pos': [],
             'language': []}
    for qid, did, query_tok, doc_tok, \
        first_bpe_positions, retriever_toks, lang in _iter_train_pairs(model, dataset, train_pairs, qrels, topic_only,
                                                                       sense_aware):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['doc_tok'].append(doc_tok)
        batch['bpe_pos'].append(first_bpe_positions)
        batch['retriever_tok'].append(retriever_toks)
        batch['language'].append(lang)
        if len(batch['query_id']) // 2 == batch_size:
            yield _pack_n_ship(batch, PACK_QLEN, PACK_DOC_LEN)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'retriever_tok': [], 'doc_tok': [], 'bpe_pos': [],
                     'language': []}


def _iter_train_pairs(model, dataset, train_pairs, qrels, topic_only=None, sense_aware=False):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                # tqdm.write("no positive labels for query %s " % qid)
                continue
            pos_id = random.choice(pos_ids)
            pos_ids_lookup = set(pos_ids)
            pos_ids = set(pos_ids)
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                tqdm.write("no negative labels for query %s " % qid)
                continue
            neg_id = random.choice(neg_ids)
            lang = qid.split("_")[1] if len(qid.split("_")) > 2 else "en"
            query_tok, first_bpe_positions, retriever_toks = model.tokenize(ds_queries[qid],
                                                                            topic=topic_only[
                                                                                qid] if topic_only is not None else None,
                                                                            lang=lang,
                                                                            clean=sense_aware)
            pos_doc = ds_docs.get(pos_id)
            neg_doc = ds_docs.get(neg_id)
            if pos_doc is None:
                tqdm.write(f'missing doc {pos_id}! Skipping')
                continue
            if neg_doc is None:
                tqdm.write(f'missing doc {neg_id}! Skipping')
                continue
            yield qid, pos_id, query_tok, model.tokenize(pos_doc)[0], first_bpe_positions, retriever_toks, lang
            yield qid, neg_id, query_tok, model.tokenize(neg_doc)[0], first_bpe_positions, retriever_toks, lang


def iter_valid_records(model, dataset, run, topic_only, batch_size, PACK_QLEN, PACK_DOC_LEN, sense_aware):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'retriever_tok': [], 'doc_tok': [], 'bpe_pos': [],
             'language': []}
    for qid, did, query_tok, doc_tok, \
        first_bpe_positions, retriever_toks, query_languages in _iter_valid_records(model, dataset, run, topic_only,
                                                                                    sense_aware):
        batch['query_id'].append(qid)
        batch['doc_id'].append(did)
        batch['query_tok'].append(query_tok)
        batch['retriever_tok'].append(retriever_toks)
        batch['doc_tok'].append(doc_tok)
        batch['bpe_pos'].append(first_bpe_positions)
        batch['language'].append(query_languages)

        if len(batch['query_id']) == batch_size:
            yield _pack_n_ship(batch, PACK_QLEN, PACK_DOC_LEN)
            batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'retriever_tok': [], 'doc_tok': [], 'bpe_pos': [],
                     'language': []}
    # final batch
    if len(batch['query_id']) > 0:
        yield _pack_n_ship(batch, PACK_QLEN, PACK_DOC_LEN)


def _iter_valid_records(model, dataset, run, topic_only=None, sense_aware=False):
    ds_queries, ds_docs = dataset
    for qid in run:
        lang = qid.split("_")[1] if len(qid.split("_")) > 2 else "en"
        query_tok, first_bpe_tokens, retriever_toks = model.tokenize(ds_queries[qid],
                                                                     lang=lang,
                                                                     clean=sense_aware,
                                                                     topic=topic_only[
                                                                         qid] if topic_only is not None else None)
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc)[0]
            yield qid, did, query_tok, doc_tok, first_bpe_tokens, retriever_toks, lang


def _pack_n_ship(batch, PACK_QLEN, PACK_DOC_LEN):
    QLEN = min(PACK_QLEN, max(len(b) for b in batch['query_tok'])) if PACK_QLEN != 20 else PACK_QLEN
    DLEN = min(PACK_DOC_LEN, max(len(b) for b in batch['doc_tok']))

    if 'retriever_tok' in batch and batch['retriever_tok'][0] is not None:
        retriever_tok_len = min(100, max(len(b) for b in batch['retriever_tok']))
        padded_retriever_tok = _pad_crop(batch['retriever_tok'], retriever_tok_len)
        padded_retriever_tok_mask = _mask(batch['retriever_tok'], retriever_tok_len)
    else:
        padded_retriever_tok = [1]
        padded_retriever_tok_mask = [1]

    padded_bpe_pos = [1]
    if 'bpe_pos' in batch and batch['bpe_pos'][0] is not None:
        bpe_pad_len = min(QLEN, max(len(b) for b in batch['bpe_pos']))
        if type(batch['bpe_pos'][0][0]) == list:
            second = max(len(el) for el in batch['bpe_pos'])
            third = max(len(ex) for el in batch['bpe_pos'] for ex in el)
            padded = list()
            for el in batch['bpe_pos']:
                pad_ex = [-1] * third
                while len(el) < second:
                    el.append(pad_ex)
                for ex in el:
                    while len(ex) < third:
                        ex.append(-1)
                padded.append(el)
            padded_bpe_pos = padded
        else:
            padded_bpe_pos = _pad_crop(batch['bpe_pos'], bpe_pad_len)

    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
        'bpe_pos': torch.tensor(padded_bpe_pos).long(),
        'retriever_tok': torch.tensor(padded_retriever_tok).long(),
        'retriever_tok_mask': torch.tensor(padded_retriever_tok_mask).long(),
        'language': batch['language'],
    }


def _pad_crop(items, l, pad_tok=-1):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [pad_tok] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    return torch.tensor(result).long()


def _mask(items, l):
    result = []
    for item in items:
        # needs padding (masked)
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        # no padding (possible crop)
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    return torch.tensor(result).float()
