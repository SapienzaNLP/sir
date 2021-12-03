import os
import pickle as pkl
from pytools import memoize_method
import torch
import numpy as np
import torch.nn.functional as tfunc
from src import modeling_util
from transformers import BertTokenizer, RagRetriever
import pytorch_pretrained_bert
import stopwordsiso as stopwords
import string
from src.data import _pad_crop, _mask

# from truecase import TrueCaser

extra_stopwords = {"en": {},
                   "it": {"Trova", "documenti", "definizione", "rapporti", "informazioni", "esempi", "parlano",
                          "informe"},
                   "es": {"Encontrar", "Busque", "papel", "documentos", "noticias", "informes", "informens", "¿",
                          "información"},
                   "fr": {"Toute", "Trouvez", "Trouver", "documents", "rapports", "informations", "Rechercher",
                          "détails", "información", "remporté"},
                   "de": {"Berichte", "Suche", "Dokumente", "Welche", "Informationen", "detaillierte"}}

true_casers = {}


# {"it": TrueCaser(dist_file_path="data/tc_models/it.dist")}#,"en": TrueCaser(dist_file_path="data/tc_models/en.dist")}


class SenseTransformerRanker(torch.nn.Module):
    def __init__(self, model_name,
                 model_size=768,
                 n_docs_retrieved=2,
                 n_docs_merged=3,
                 retriever_name="bert-large-cased",
                 normalize_query=True,
                 frozen_query_encoder=True,
                 log_retrieval=None):
        super().__init__()
        self.MODEL = model_name  # 'bert-base-uncased'
        self.MODEL_SIZE = model_size  # 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased

        self.bert = CustomBertModel.from_pretrained(self.MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL)
        print("Finished bert model init!")
        if retriever_name is None or 'multilingual' in model_name:
            retriever_name = model_name
        print(retriever_name, "<--- Retriever model")
        if retriever_name == "bert-base-multilingual-cased":
            dataset_path = "data/ares/normalized_faiss_index_no_stopsynsets_multilingual_exact_match"
            index_path = os.path.join(dataset_path, "ares_hnsw_index.faiss")
        else:
            # dataset saved via `dataset.save_to_disk(...)`
            dataset_path = "data/ares/ares_bert_large_faiss_index_exact"
            # faiss index saved via `dataset.get_index("embeddings").save(...)`
            index_path = os.path.join(dataset_path, "ares_l2_index.faiss")

        self.retriever_tokenizer = BertTokenizer.from_pretrained(retriever_name)
        doc_sep = " {} ".format(self.retriever_tokenizer.sep_token)
        self.retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom',
                                                      passages_path=dataset_path, index_path=index_path, title_sep="",
                                                      doc_sep=doc_sep)
        self.retriever.question_encoder_tokenizer = self.retriever_tokenizer
        self.retriever.generator_tokenizer = self.retriever_tokenizer
        self.frozen_query_encoder = frozen_query_encoder
        print("Finished rag model init!")
        if self.frozen_query_encoder:
            self.query_encoder = CustomBertModel.from_pretrained(retriever_name).eval()
        else:
            self.query_encoder = CustomBertModel.from_pretrained(retriever_name)

        self.normalize_query = normalize_query
        self.n_docs_retrieved = n_docs_retrieved
        self.n_docs_merged = n_docs_merged

        self.log_retrieval = log_retrieval
        if log_retrieval is not None:
            if not frozen_query_encoder:
                self.log_retrieval_txt = open(log_retrieval + ".retrieval", "w")
                self.log_retrieval_dict = dict()
            else:
                self.log_retrieval_cache = log_retrieval + ".retrieval.pkl"
                self.log_retrieval_txt = open(log_retrieval + ".retrieval", "w")
                if os.path.exists(self.log_retrieval_cache):
                    print("Loading previously computed cache!")
                    self.log_retrieval_dict = pkl.load(open(self.log_retrieval_cache, "rb"))
                else:
                    self.log_retrieval_dict = dict()
        else:
            self.log_retrieval_dict = dict()

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def clean_bpes(self, text, topic=None, lang=None, clean=False, lower_case=False, true_case=False):
        if lang == "it":
            true_case = True
        tokenizer = self.retriever_tokenizer if self.retriever_tokenizer is not None else self.tokenizer
        if lower_case:
            text = text.lower()
            if topic is not None:
                topic = topic.lower()
        if true_case and lang in true_casers:
            text = true_casers[lang].get_true_case(text, "lower")
            if topic is not None:
                topic = true_casers[lang].get_true_case(topic, "lower")
        # BE CAREFUL, might work for BERT tokenizer only
        split_tokens = list()
        stopword_subtokens_pos = list()
        topic_subtokens_pos = list()
        topic_split_tokens = list()

        if clean:
            # toks_prev = self.tokenizer.tokenize(text)
            for token in tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens):
                token_pieces = tokenizer.wordpiece_tokenizer.tokenize(token)
                if token in stopwords.stopwords(lang) \
                        or token.lower() in stopwords.stopwords(lang) \
                        or token in extra_stopwords[lang] or token in string.punctuation:
                    for p in range(len(split_tokens), len(split_tokens) + len(token_pieces)):
                        stopword_subtokens_pos.append(p)
                split_tokens += token_pieces
            toks = split_tokens

            if topic is not None:
                for token in tokenizer.basic_tokenizer.tokenize(topic, never_split=tokenizer.all_special_tokens):
                    token_pbes = list()
                    token_pieces = tokenizer.wordpiece_tokenizer.tokenize(token)
                    if token not in stopwords.stopwords(lang) \
                            and not token.lower() in stopwords.stopwords(lang) \
                            and token not in extra_stopwords[lang] and token not in string.punctuation:
                        for p in range(len(topic_split_tokens), len(topic_split_tokens) + len(token_pieces)):
                            token_pbes.append(p + 1)
                    if len(token_pbes) > 0:
                        topic_subtokens_pos.append(token_pbes)
                    topic_split_tokens += token_pieces
                first_bpe_positions = topic_subtokens_pos
            else:
                first_bpe_positions = [i + 1 for i, bpe in enumerate(toks) if
                                       not bpe.startswith("##") and i not in stopword_subtokens_pos]

        else:
            toks = self.tokenizer.tokenize(text)
            first_bpe_positions = [i + 1 for i, bpe in enumerate(toks) if not bpe.startswith("##")]
        # pos+1 is needed coz later we add the cls token to be first
        return toks, first_bpe_positions

    @memoize_method
    def tokenize(self, text, topic=None, lang=None, clean=False):
        retriever_toks = None
        first_bpe_positions = None
        if self.retriever_tokenizer is not None and clean:
            retriever_toks_, first_bpe_positions = self.clean_bpes(text, topic=topic, lang=lang, clean=clean)
            retriever_toks = [self.retriever_tokenizer.vocab[t] for t in retriever_toks_]
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks, first_bpe_positions, retriever_toks

    @staticmethod
    def extract_output_bpe(toexpand_query_encode, bpe_pos, normalize=False):
        # build first BPE per word representation
        mean_vecs = torch.sum(torch.stack(toexpand_query_encode[-4:], dim=0), dim=0)
        # duplicate the cls vector to be compatible with ARES
        mean_vecs = torch.cat((mean_vecs, mean_vecs), dim=-1)
        batch_bpe_vecs = list()
        if len(bpe_pos.shape) < 3:
            for i, example in enumerate(mean_vecs):
                query_bpe_repr = example[[x.item() for x in bpe_pos[i] if x != -1]]
                if normalize:
                    query_bpe_repr = tfunc.normalize(query_bpe_repr, p=2, dim=-1)
                batch_bpe_vecs.append(query_bpe_repr)
        else:
            for i, example in enumerate(mean_vecs):
                query_vecs = []
                for token_bpes in bpe_pos[i]:
                    token_vec = example[[x.item() for x in token_bpes if x.item() != -1]]
                    if token_vec.shape[0] > 0:
                        query_vecs.append(torch.mean(token_vec, dim=0))
                query_bpe_repr = torch.stack(query_vecs)
                if normalize:
                    query_bpe_repr = tfunc.normalize(query_bpe_repr, p=2, dim=-1)
                batch_bpe_vecs.append(query_bpe_repr)
        return batch_bpe_vecs

    def extract_top_bpe_glosses(self, retrieval_query_toks, bpe_pos, doc_scores, retrieved_doc_ids, original_query_toks,
                                query_language=None, use_english_only=False):

        condition = (torch.where(original_query_toks == -1)[0]).tolist()
        gloss_scores = doc_scores.view(1, -1).tolist()[0]
        retrievals = self.retriever.index.get_doc_dicts(retrieved_doc_ids)
        glosses = []
        for r in retrievals:
            if 'gloss' in retrievals[0] and query_language is None:
                glosses.append(r['gloss'])
            elif query_language is not None and query_language.upper() in r:
                if r[query_language.upper()] != ["NONE"] and not use_english_only:
                    glosses.extend(r[query_language.upper()])
                else:
                    glosses.extend(r['EN'])
            elif 'gloss' in retrievals[0]:
                glosses.extend(r['gloss'])

        srt_scores, srt_glosses = (list(t) for t in zip(*sorted(zip(gloss_scores, glosses), reverse=True)))
        selected_bpe_indices_all = (-np.array(gloss_scores)).argsort().tolist()
        bpe_glosses = self.bpepos2bpe(bpe_pos, selected_bpe_indices_all, retrieval_query_toks)
        unique_glosses = list()
        unique_gloss_scores = list()
        # remove duplicates
        for m, gloss in enumerate(srt_glosses):
            if len(unique_glosses) == self.n_docs_merged:
                break
            if gloss not in set(unique_glosses):
                unique_glosses.append(gloss)
                unique_gloss_scores.append(srt_scores[m])

        toappend_glosses = " / ".join(unique_glosses)  # join glosses by /
        tokenized_glosses = self.tokenizer.tokenize(toappend_glosses)
        glosses_toks = [self.tokenizer.vocab[t] for t in tokenized_glosses][:100]
        end = condition[0] if len(condition) > 0 else len(original_query_toks)
        expanded_query = glosses_toks + [self.tokenizer.vocab[":"]] + original_query_toks[:end].tolist()

        return expanded_query, unique_glosses, unique_gloss_scores, bpe_glosses

    def bpepos2bpe(self, bpe_pos, selected_bpe_indices_all, query_toks):
        selected_bpes = bpe_pos[selected_bpe_indices_all]
        tokens = []
        for tok in selected_bpes:
            tok_bpe = []
            for idx in tok:
                if idx == -1:
                    continue
                tok_bpe.append(query_toks[idx])
            tokens.append("".join(self.retriever_tokenizer.convert_ids_to_tokens(tok_bpe)))
        return tokens

    def retrieve_for_token(self, toks, outputs, bpe_pos, original_query_toks, query_languages=None,
                           batch_retrievals=None, log_valid=False, use_english_only=False):
        # This is a list of matrices for each example in batch
        expanded_queries = list()
        token_doc_scores = list()
        query_glosses = list()
        toexpand_query_encode_bpe = self.extract_output_bpe(outputs, bpe_pos, normalize=self.normalize_query)
        batch_retrievals_return = list()
        batch_bpe_retrievals_return = list()
        for query_idx, bpes in enumerate(toexpand_query_encode_bpe):
            if batch_retrievals is None:
                retriever_outputs = self.retriever(
                    toks[query_idx].tolist(),
                    bpes.cpu().detach().to(torch.float32).numpy(),
                    n_docs=self.n_docs_retrieved,
                    return_tensors="pt",
                )
            else:
                retriever_outputs = batch_retrievals[query_idx]

            retrieved_doc_embeds = retriever_outputs["retrieved_doc_embeds"]
            retrieved_doc_ids = retriever_outputs["doc_ids"]

            retrieved_doc_embeds = retrieved_doc_embeds.to(bpes)
            doc_scores = torch.bmm(bpes.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            expanded_query, toappend_glosses, glosses_scores, selected_bpe_glosses = self.extract_top_bpe_glosses(
                toks[query_idx],
                bpe_pos[query_idx],
                doc_scores,
                retrieved_doc_ids,
                original_query_toks[query_idx],
                query_language=query_languages[query_idx],
                use_english_only=use_english_only)
            query_glosses.append((toappend_glosses, glosses_scores))
            expanded_queries.append(expanded_query)
            token_doc_scores.append(doc_scores)
            batch_retrievals_return.append(retriever_outputs)
            batch_bpe_retrievals_return.append(selected_bpe_glosses)

        query_len = max([len(q) for q in expanded_queries])
        final_expanded_query_tok, final_query_mask = _pad_crop(expanded_queries, query_len).to(toks), _mask(
            expanded_queries, query_len).to(toks)

        if self.log_retrieval is not None and log_valid:  # let this be an open file
            for qidx, query_gloss in enumerate(query_glosses):
                qg = query_gloss[0]
                sg = query_gloss[1]
                bpe = batch_bpe_retrievals_return[qidx]
                # bpe = "@"
                r = min(self.n_docs_merged, len(qg))
                if r > 0:
                    # gloss_score = [str(sg[i])+"-"+bpe[i]+"-"+qg[i] for i in range(r)]
                    gloss_score = [str(sg[i]) + "-" + bpe[i] + "-" + qg[i] for i in range(r)]
                else:
                    gloss_score = []
                self.log_retrieval_txt.write("{}\n".format("\t".join(gloss_score)))

        return final_expanded_query_tok, final_query_mask, token_doc_scores, \
               batch_bpe_retrievals_return, batch_retrievals_return

    def expand_queries(self, query_toks, query_mask, bpe_pos, original_query_toks, query_languages, batch_id=None,
                       use_english_only=False):
        BATCH, query_len = query_toks.shape
        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build query BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + query_len), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)
        if self.frozen_query_encoder:
            self.query_encoder.eval()
            with torch.no_grad():
                outputs = self.query_encoder(input_ids=toks.long(), token_type_ids=segment_ids.long(),
                                             attention_mask=mask.long())
        else:
            outputs = self.query_encoder(input_ids=toks.long(), token_type_ids=segment_ids.long(),
                                         attention_mask=mask.long())

        if not self.frozen_query_encoder:
            batch_retrievals = None
        elif batch_id is None or batch_id not in self.log_retrieval_dict:
            batch_retrievals = None
        else:
            batch_retrievals, _ = self.log_retrieval_dict[batch_id]

        final_expanded_query_tok, final_query_mask, doc_scores, batch_bpe_retrievals_return, batch_retrievals_return = \
            self.retrieve_for_token(toks, outputs, bpe_pos, original_query_toks, query_languages, batch_retrievals,
                                    log_valid=batch_id is not None, use_english_only=use_english_only)
        if self.frozen_query_encoder:
            if batch_id is not None and self.log_retrieval is not None:
                self.log_retrieval_dict[batch_id] = (batch_retrievals_return, batch_bpe_retrievals_return)

        return final_expanded_query_tok, final_query_mask, doc_scores

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, retriever_tok=None, retriever_tok_mask=None,
                    bpe_pos=None, query_languages=None, batch_id=None, use_english_only=False):
        query_tok, query_mask, gloss_scores = self.expand_queries(
            retriever_tok if retriever_tok is not None else query_tok,
            retriever_tok_mask if retriever_tok_mask is not None else query_mask,
            bpe_pos,
            original_query_toks=query_tok,
            query_languages=query_languages,
            batch_id=batch_id,
            use_english_only=use_english_only)

        BATCH, query_len = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - query_len - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + query_len) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(input_ids=toks, token_type_ids=segment_ids.long(), attention_mask=mask)
        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:query_len + 1] for r in result]
        doc_results = [r[:, query_len + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaSenseRanker(SenseTransformerRanker):
    def __init__(self, model_name, sense_aware_args, log_retrieval=None):
        super().__init__(model_name=model_name,
                         n_docs_retrieved=sense_aware_args.n_docs,
                         n_docs_merged=sense_aware_args.n_merged,
                         normalize_query=sense_aware_args.normalize_query,
                         frozen_query_encoder=sense_aware_args.freeze_query_encoder,
                         log_retrieval=log_retrieval)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.MODEL_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, bpe_pos=None, retriever_tok=None,
                retriever_tok_mask=None, query_languages=None, batch_id=None, use_english_only=False):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask,
                                          retriever_tok=retriever_tok,
                                          retriever_tok_mask=retriever_tok_mask,
                                          bpe_pos=bpe_pos,
                                          query_languages=query_languages,
                                          batch_id=batch_id,
                                          use_english_only=use_english_only)
        return self.cls(self.dropout(cls_reps[-1]))


class BertRanker(torch.nn.Module):
    def __init__(self, bert_name):
        super().__init__()
        self.BERT_MODEL = bert_name  # 'bert-base-uncased'
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def extract_output(self, outputs, normalize=False):
        hidden_states = outputs[2]
        embedding_output = hidden_states[0]
        encoded_layers = hidden_states[1:]
        return [embedding_output] + list(encoded_layers), None

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    @memoize_method
    def tokenize(self, text, topic=None, lang=None, clean=False):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks, None, None

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, query_len = query_tok.shape
        DIFF = 3  # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - query_len - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + query_len) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0  # remove padding (will be masked anyway)

        # execute BERT model
        result = self.bert(toks, segment_ids.long(), mask)

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:query_len + 1] for r in result]
        doc_results = [r[:, query_len + 2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i * BATCH:(i + 1) * BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class VanillaBertRanker(BertRanker):
    def __init__(self, bert_name):
        super().__init__(bert_name=bert_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, bpe_pos=None,
                retriever_tok=None, retriever_tok_mask=None, query_languages=None, batch_id=None,
                use_english_only=False):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        return self.cls(self.dropout(cls_reps[-1]))


class CustomBertModel(pytorch_pretrained_bert.BertModel):  # pytorch_pretrained_bert.
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        """
        Based on pytorch_pretrained_bert.BertModel
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=True)

        return [embedding_output] + encoded_layers
