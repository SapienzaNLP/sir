import os
import re
import math
import random
import pickle as pkl
from tqdm import tqdm
# from lxml import html as etree
from collections import defaultdict
from xml.etree import ElementTree as et

random.seed(42)


def flatten(lst):
    return [_e for sub_l in lst for _e in sub_l]


def clef_data_tables():
    clef_langs = "de es fr it ru en pt".split()
    mono_clef_per_year = {}
    bi_clef_per_year = {}

    for lang in clef_langs:
        mono_clef_per_year[lang] = {}
        bi_clef_per_year[lang] = {}
        for year in ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007']:
            mono_clef_per_year[lang][year] = dict()
            mono_clef_per_year[lang][year]['collections'] = list()
            mono_clef_per_year[lang][year]['targets'] = [lang]
            mono_clef_per_year[lang][year]['sources'] = [lang]

            bi_clef_per_year[lang][year] = dict()
            bi_clef_per_year[lang][year]['collections'] = list()
            bi_clef_per_year[lang][year]['targets'] = [lang]
            bi_clef_per_year[lang][year]['sources'] = list()
    '''
    TODO complete multilingual task
    multi_clef_per_year = {}
    for year in ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007']:
        multi_clef_per_year[year] = dict()
        multi_clef_per_year[year]['collections'] = list()
        multi_clef_per_year[year]['targets'] = list()
        multi_clef_per_year[year]['source'] = list()
    '''
    # AH monolingual task
    mono_clef_per_year['de']['2000']['collections'].extend([('FRANKFURTER1994', 'FR94')])
    mono_clef_per_year['de']['2001']['collections'].extend([('FRANKFURTER1994', 'FR94')])
    mono_clef_per_year['de']['2002']['collections'].extend([('SDA1994', 'SDA.94')])
    mono_clef_per_year['de']['2003']['collections'].extend(
        [('SPIEGEL1994', 'SPIEGEL9495'), ('SPIEGEL1995', 'SPIEGEL9495')])

    mono_clef_per_year['es']['2001']['collections'].extend([('EFE1994', 'EFE1994')])
    mono_clef_per_year['es']['2002']['collections'].extend([('EFE1994', 'EFE1994')])
    mono_clef_per_year['es']['2003']['collections'].extend([('EFE1994', 'EFE1994'), ('EFE1995', 'EFE1995')])

    mono_clef_per_year['fr']['2000']['collections'].extend([('LEMONDE1994', 'LEMONDE94')])
    mono_clef_per_year['fr']['2001']['collections'].extend([('LEMONDE1994', 'LEMONDE94'), ('ATS1994', 'ATS.94')])
    mono_clef_per_year['fr']['2002']['collections'].extend([('LEMONDE1994', 'LEMONDE94'), ('ATS1994', 'ATS.94')])
    mono_clef_per_year['fr']['2003']['collections'].extend(
        [('LEMONDE1994', 'LEMONDE94'), ('ATS1994', 'ATS.94'), ('ATS1995', 'ATS.95')])
    mono_clef_per_year['fr']['2004']['collections'].extend([('LEMONDE1995', 'LEMONDE95'), ('ATS1995', 'ATS.95')])
    mono_clef_per_year['fr']['2005']['collections'].extend(
        [('LEMONDE1994', 'LEMONDE94'), ('ATS1994', 'ATS.94'), ('LEMONDE1995', 'LEMONDE95'), ('ATS1995', 'ATS.95')])
    mono_clef_per_year['fr']['2006']['collections'].extend(
        [('LEMONDE1994', 'LEMONDE94'), ('ATS1994', 'ATS.94'), ('LEMONDE1995', 'LEMONDE95'), ('ATS1995', 'ATS.95')])

    mono_clef_per_year['it']['2000']['collections'].extend([('AGZ1994', 'AGZ.94'), ('LASTAMPA1994', 'LASTAMPA94')])
    mono_clef_per_year['it']['2001']['collections'].extend([('AGZ1994', 'AGZ.94'), ('LASTAMPA1994', 'LASTAMPA94')])
    mono_clef_per_year['it']['2002']['collections'].extend([('AGZ1994', 'AGZ.94'), ('LASTAMPA1994', 'LASTAMPA94')])
    mono_clef_per_year['it']['2003']['collections'].extend(
        [('AGZ1994', 'AGZ.94'), ('AGZ1995', 'AGZ.95'), ('LASTAMPA1994', 'LASTAMPA94')])

    # AH bilingual task X2{DE,EN,ES,FR,IT,RU}
    bi_clef_per_year['de']['2000']['collections'].extend(['FRANKFURTER1994'])
    bi_clef_per_year['de']['2000']['sources'].extend(['en', 'fr', 'ru'])

    bi_clef_per_year['en']['2000']['collections'].extend(['LATIMES1994'])
    bi_clef_per_year['en']['2000']['sources'].extend(['de', 'es', 'fr', 'it'])
    bi_clef_per_year['en']['2001']['collections'].extend(['LATIMES1994'])
    bi_clef_per_year['en']['2001']['sources'].extend(['de', 'es', 'fr', 'it', 'ja', 'ru', 'zh'])
    bi_clef_per_year['en']['2002']['collections'].extend(['LATIMES1994'])
    bi_clef_per_year['en']['2002']['sources'].extend(['es', 'fr', 'pt', 'zh'])
    bi_clef_per_year['en']['2003']['collections'].extend(['LATIMES1994', 'GLASGOW1995'])
    bi_clef_per_year['en']['2003']['sources'].extend(['de', 'es', 'fr', 'it'])
    bi_clef_per_year['en']['2004']['collections'].extend(['LATIMES1994', 'GLASGOW1995'])
    bi_clef_per_year['en']['2004']['sources'].extend(['es', 'fr'])
    bi_clef_per_year['en']['2005']['collections'].extend(['LATIMES1994', 'GLASGOW1995'])
    bi_clef_per_year['en']['2005']['sources'].extend(['en', 'ru'])
    bi_clef_per_year['en']['2006']['collections'].extend(['LATIMES1994', 'GLASGOW1995'])
    bi_clef_per_year['en']['2006']['sources'].extend(['it'])
    bi_clef_per_year['en']['2007']['collections'].extend(['LATIMES2002'])
    bi_clef_per_year['en']['2007']['sources'].extend(['zh'])

    bi_clef_per_year['es']['2002']['collections'].extend(['EFE1994'])
    bi_clef_per_year['es']['2002']['sources'].extend(['de', 'en', 'fr', 'it', 'pt'])
    bi_clef_per_year['es']['2003']['collections'].extend(['EFE1994', 'EFE1995'])
    bi_clef_per_year['es']['2003']['sources'].extend(['it'])

    bi_clef_per_year['fr']['2002']['collections'].extend(['LEMONDE1994', 'ATS1994'])
    bi_clef_per_year['fr']['2002']['sources'].extend(['de', 'en', 'ru'])
    bi_clef_per_year['fr']['2004']['collections'].extend(['LEMONDE1995', 'ATS1995'])
    bi_clef_per_year['fr']['2004']['sources'].extend(['de'])
    bi_clef_per_year['fr']['2005']['collections'].extend(['LEMONDE1994', 'ATS1994', 'LEMONDE1995', 'ATS1995'])
    bi_clef_per_year['fr']['2005']['sources'].extend(['de', 'en', 'ru', 'es', 'it'])

    bi_clef_per_year['it']['2002']['collections'].extend(['AGZ1994', 'LASTAMPA1994'])
    bi_clef_per_year['it']['2002']['sources'].extend(['de', 'en', 'fr', 'es'])
    bi_clef_per_year['it']['2003']['collections'].extend(['AGZ1994', 'AGZ1995', 'LASTAMPA1994'])
    bi_clef_per_year['it']['2003']['sources'].extend(['de'])

    bi_clef_per_year['pt']['2004']['collections'].extend(['PUBLICO1994', 'PUBLICO1995'])
    bi_clef_per_year['pt']['2004']['sources'].extend(['en', 'es'])
    bi_clef_per_year['pt']['2005']['collections'].extend(['PUBLICO1994', 'PUBLICO1995', 'FOLHA1994', 'FOLHA1995'])
    bi_clef_per_year['pt']['2005']['sources'].extend(['en', 'es', 'fr'])
    bi_clef_per_year['pt']['2006']['collections'].extend(['PUBLICO1994', 'PUBLICO1995', 'FOLHA1994', 'FOLHA1995'])
    bi_clef_per_year['pt']['2006']['sources'].extend(['en', 'es', 'fr'])

    bi_clef_per_year['ru']['2003']['collections'].extend(['IZVESTIA1995'])
    bi_clef_per_year['ru']['2003']['sources'].extend(['en', 'de'])
    bi_clef_per_year['ru']['2004']['collections'].extend(['IZVESTIA1995'])
    bi_clef_per_year['ru']['2004']['sources'].extend(['en', 'es', 'fr', 'ja', 'zh'])

    # AH multilingual task X2X

    return mono_clef_per_year, bi_clef_per_year


def cedr_query_format(data_dir, query_portion, test_benchmark):
    queries = os.path.join(data_dir, "queries",
                           "title-desc_as_query.tsv" if query_portion == "both" else "title_as_query.tsv")
    pool = os.path.join(data_dir, "qrels", test_benchmark + "_qrels.tsv")
    folds = os.path.join(data_dir, "folds", query_portion)
    query2relevant = dict()
    with open(pool, "r") as infile:
        for line in infile:
            query_id, _, doc_id, relevance = line.rstrip().split()
            if query_id not in query2relevant:
                query2relevant[query_id] = set()
            if int(relevance) == 1:
                query2relevant[query_id].add(doc_id)

    query2docs = dict()
    for file in os.listdir(folds):
        if not file.endswith(".dev.run"):
            continue
        with open(os.path.join(folds, file)) as infile:
            for line in infile:
                query_id, _, doc_id, _, score, _ = line.rstrip().split()
                if query_id not in query2docs:
                    query2docs[query_id] = dict()
                    query2docs[query_id]['relevant_docs'] = list()
                    query2docs[query_id]['irrelevant_docs'] = list()
                current_dict = dict()
                current_dict['title'] = current_dict['indexed_id'] = doc_id
                current_dict['corpora'] = 'clef'
                if doc_id in query2relevant[query_id]:
                    current_dict['score'] = 1
                    current_dict['discrete_score'] = 1
                    query2docs[query_id]['relevant_docs'].append(current_dict)
                else:
                    current_dict['score'] = 0
                    current_dict['discrete_score'] = 0
                    query2docs[query_id]['irrelevant_docs'].append(current_dict)

    query_jsons = list()
    with open(queries, "r") as infile:
        for line in infile:
            fields = line.rstrip().split("\t")
            query_dict = dict()
            query_id = fields[1]
            if query_id not in query2docs: continue
            query_dict['query'] = fields[2]
            query_dict['query_id'] = query_id
            query_dict['relevant_docs'] = query2docs[query_id]['relevant_docs']
            query_dict['irrelevant_docs'] = query2docs[query_id]['irrelevant_docs']
            query_dict['corpora'] = 'clef'
            query_jsons.append(query_dict)

    return query_jsons


def clef_query_format(year, lang, relevant_docs=None, pool=None, task="MONO", code2lang=None,
                      parts="both"):  # task can be {"MONO","BILI", "MULTI}
    root_dir = "data/clef_data/topics/AH-CLEF{}/topics".format(year)
    if task == "BILI":
        filename = "{}_topics_AH-{}-X2EN-CLEF{}.xml".format(code2lang[lang], task, year)
    elif task == "MONO":
        filename = "{}_topics_AH-{}-{}-CLEF{}.xml".format(code2lang[lang], task, lang.upper(), year)
    elif task == "MULTI":
        filename = "{}_topics_AH-{}-CLEF{}.xml".format(code2lang[lang], task, year)
        raise NotImplementedError
    else:
        raise ValueError

    queries, ids, full_ids = load_topics(os.path.join(root_dir, filename))
    if pool is not None:
        query2docs = load_pools(pool)
        query_jsons = list()
        for i, query in enumerate(queries):
            query_dict = dict()
            title, desc = query
            query_id = ids[i]
            if parts == "title":
                query_dict['query'] = title
            else:
                query_dict['query'] = " ".join([title, desc])
            query_dict['query_id'] = query_id
            query_dict['relevant_docs'] = query2docs[query_id]['relevant_docs']
            query_dict['irrelevant_docs'] = query2docs[query_id]['irrelevant_docs']
            query_dict['corpora'] = 'clef'
            query_jsons.append(query_dict)
    elif relevant_docs is not None:
        relevant_docs_list = []
        for el in relevant_docs:
            current_dict = dict()
            current_dict['title'] = el
            current_dict['score'] = 999
            current_dict['discrete_score'] = 999
            current_dict['indexed_id'] = el
            current_dict['corpora'] = 'clef'
            relevant_docs_list.append(current_dict)

        query_jsons = list()
        for i, query in enumerate(queries):
            query_dict = dict()
            title, desc = query
            query_id = ids[i]
            query_dict['query'] = title
            query_dict['query_id'] = query_id
            query_dict['relevant_docs'] = relevant_docs_list
            query_dict['irrelevant_docs'] = []
            query_dict['corpora'] = 'clef'
            query_jsons.append(query_dict)
    else:
        raise ValueError

    return query_jsons


def load_topics(path):
    queries = []
    ids = []
    full_ids = []
    tree = et.parse(path)
    root = tree.getroot()
    all_topics = root.findall("topic")
    for i, topic in enumerate(all_topics):
        _identifier = topic.find("identifier").text.strip()  # e.g. 'C041'
        _id = _identifier.split("-")[0]  # e.g. 41
        title = topic.find("title").text.strip()
        desc = topic.find("description").text.strip()
        all_context = [title, desc]
        ids.append(_id)
        queries.append(all_context)
        full_ids.append(_identifier)

    return queries, ids, full_ids


def load_pools(path, total_docs=150):
    random.seed(42)
    # "10 0 FR941211-000894 0"
    query2docs = dict()
    with open(path, "r") as infile:
        for line in infile:
            if len(line.split()) < 4:
                continue
            query_id, _, doc_id, rel = line.split()
            if query_id not in query2docs:
                query2docs[query_id] = dict()
                query2docs[query_id]['relevant_docs'] = list()
                query2docs[query_id]['irrelevant_docs'] = list()
                query2docs[query_id]['corpora'] = "clef"
            if rel == "0":
                query2docs[query_id]['irrelevant_docs'].append(doc_id)
            if rel == "1":
                query2docs[query_id]['relevant_docs'].append(doc_id)

    for query_id in query2docs:
        relevant_docs_list = []
        for el in query2docs[query_id]['relevant_docs']:
            current_dict = dict()
            current_dict['title'] = el
            current_dict['score'] = 1
            current_dict['discrete_score'] = 1
            current_dict['indexed_id'] = el
            current_dict['corpora'] = 'clef'
            relevant_docs_list.append(current_dict)
        query2docs[query_id]['relevant_docs'] = relevant_docs_list

        to_sample = 150 - len(query2docs[query_id]['relevant_docs'])
        irrel = random.sample(query2docs[query_id]['irrelevant_docs'],
                              min(to_sample, len(query2docs[query_id]['irrelevant_docs'])))

        irrelevant_docs_list = list()
        for el in irrel:
            current_dict = dict()
            current_dict['title'] = el
            current_dict['score'] = 0
            current_dict['discrete_score'] = 0
            current_dict['indexed_id'] = el
            current_dict['corpora'] = 'clef'
            irrelevant_docs_list.append(current_dict)
        query2docs[query_id]['irrelevant_docs'] = irrelevant_docs_list

    return query2docs


def clean_text_n(s, corpora=None, lang=None):
    # html_escape_table
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    s = s.replace("&amp;", "&")
    s = s.replace("&quot;", '"')
    s = s.replace("&apos;", "'")
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.strip()
    s = re.sub(' +', ' ', s)
    s = s.replace("_", " ")
    # if lang is not None and corpora is not None:
    #     if lang=="es" and corpora=="clef":
    #         s = re.sub(' +', ' ', s)
    #     if lang=="en" and corpora=="wiki":
    #         s = s.replace("_", " ")

    return s


def format_scores(results):
    all_pairs = list()
    all_scores = list()
    rerank_run = defaultdict(dict)
    for el in results:
        all_pairs.extend(el[0])
        if type(el[1]) == list:
            all_scores.extend(el[1])
        else:
            all_scores.append(el[1])

    for i, pk in enumerate(all_pairs):
        score = all_scores[i]
        # 'it::clef::1--clef::AGZ.940520.0107-0'
        pair = all_pairs[i].split("--")
        topic_id = pair[0].split("::")[-1]
        doc_id = pair[1].split("::")[-1]  # [:-2]
        rerank_run[topic_id][doc_id] = score

    return rerank_run


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_collection_keys(document_root, docs, language, year, test_benchmark):
    relevant_doc_ids = list()
    lang_year_corpora = docs[language][year]['collections']
    abbvs = [x[1] for x in lang_year_corpora]
    with open(os.path.join(document_root, test_benchmark, "{}.docid2idx.pkl".format(language)), "rb") as infile:
        docs = pkl.load(infile)
    for doc in docs:
        for abbv in abbvs:
            if abbv in doc:
                relevant_doc_ids.append(doc)
    return relevant_doc_ids


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, folds=5):
    l = ["clef" + "::" + x for x in l]
    random.shuffle(l)
    n = math.ceil(len(l) / folds)
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_qrels_dict(file):
    result = {}
    for line in tqdm(open(file, "rt"), desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def load_bn2gloss():
    bn2gloss = dict()
    gloss2bn = dict()
    with open("data/ares/bn_offset_to_gloss.txt", "r") as infile:
        for line in infile:
            fields = line.rstrip().split("\t")
            gloss = fields[1]
            bn = fields[0]
            gloss2bn[gloss] = bn
    return gloss2bn


def query2retrieved2(queries, retriveal_log):
    gloss2bn = load_bn2gloss()
    query2glosses = dict()
    for i, line in enumerate(retriveal_log.readlines()):
        fields = line.rstrip().split("\t")
        query_id = queries[i]
        if query_id not in query2glosses:
            query2glosses[query_id] = list()
        else:
            continue
        for el in fields:
            sel = el.split("-")
            gloss = "-".join(sel[2:])
            # gloss = "-".join(sel[1:])
            score = sel[0]
            bpe = sel[1]
            if gloss in gloss2bn:
                bn = gloss2bn[gloss]
            else:
                bn = "NO_BN"
            query2glosses[query_id].append("{:.2f} - {} - {} - {}".format(float(score), bpe, bn, gloss))
            # query2glosses[query_id].append("{:.2f} - {} - {}".format(float(score)*100, bn, gloss))
        # query2glosses[query_id]=fields

    return query2glosses


def postprocess_output(args, retrieval_cache):
    retrieval_log = open(retrieval_cache + ".retrieval", "r")
    query_ids = [x.rstrip().split()[0] for x in open(args.run, "r")]
    query2title = read_datafiles(args.datafiles)[0]
    query2glosses = query2retrieved2(query_ids, retrieval_log)
    with open(retrieval_cache + ".retrieval-bn.merge", "w") as outfile:
        for query in query2glosses:
            query_text = query2title[query] if query in query2title else ""
            outfile.write("{}\t{}\t{}\n".format(query, query_text, "\t".join(query2glosses[query])))


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
