import os
# import re
import json
import argparse
# import jsonlines
import subprocess
import pickle as pkl
from tqdm import tqdm
from pyserini.search import SimpleSearcher
from jnius import autoclass
from src.data_creation.data_utils import load_pools, load_topics, clef_data_tables

String = autoclass("java.lang.String")


def index_only():
    maind = "data/clef_data/clef_collection-json"
    for f in os.listdir(maind):
        cd = os.path.join(maind, f)
        if not os.path.isdir(cd):
            continue
        collection_name = f
        # if collection_name == "fr_LEMONDE95-ATS.95" or collection_name=="de_SDA.94"
        # or collection_name=="it_AGZ.94-AGZ.95-LASTAMPA94":
        subprocess.check_call(['src/cedr/scripts/index.sh', cd, collection_name])
        # break


def misc_index():
    searcher = SimpleSearcher("data/clef_data/clef_anserini_indexes/{}".format("de_SDA.94_trial"))
    doc = searcher.doc("AGZ.941231.0042")
    return


def load_documents(collection_documents, subcollection_ids):
    all_documents = dict()
    for path in tqdm(collection_documents, desc="Collections"):
        with open(path, "rb") as infile:
            all_documents.update(pkl.load(infile))
    if subcollection_ids is not None:
        filtered_documents = dict()
        for el in all_documents:
            for subid in subcollection_ids:
                if type(all_documents[el]['id']) == list:
                    for elid in all_documents[el]['id']:
                        if elid.startswith(subid):
                            filtered_documents[el] = all_documents[el]
                            break
                else:
                    if all_documents[el]['id'].startswith(subid):
                        filtered_documents[el] = all_documents[el]
                        break
        all_documents = filtered_documents
    return all_documents


def dump_json_from_collections(path, outpath, subcollections=None):
    all_documents_json = list()
    all_documents = load_documents([path], subcollection_ids=subcollections)
    for el in tqdm(all_documents, desc="Writing in json format:"):
        for i in range(len(all_documents[el]['id'])):
            current = {"id": all_documents[el]['id'][i], "contents": all_documents[el]['text'][i]}
            all_documents_json.append(current)
    with open(outpath, "w") as outfile:
        json.dump(all_documents_json, outfile)


def parse_response(response, lang, corpora):
    all_hits = list()
    for i in range(len(response)):
        current_hit = dict()
        current_hit['title'] = ""
        current_hit['score'] = response[i].score
        current_hit['indexed_id'] = response[i].docid
        all_hits.append(current_hit)
    return sorted(all_hits, key=lambda k: k['score'], reverse=True)


def single_search_test(year, lang, subcollections, topics_path, parts, max_results, out_results):
    collection_name = "{}_{}".format(lang, "-".join(subcollections))
    topics = load_topics(topics_path)
    searcher = SimpleSearcher("/media/rexhina/storage3/IR/clef_anserini_indexes/{}".format(collection_name))
    # searcher.set_bm25(k1=float(1), b=float(0.5))
    searcher.set_rm3()
    save_all = []
    for idx in range(len(topics[0])):
        title_desc = topics[0][idx]
        queryid = topics[1][idx]
        if parts == "both":
            query = " ".join(title_desc)
        else:
            query = title_desc[0]

        save_dict = dict()
        save_dict["query"] = query
        save_dict["query_id"] = queryid
        hits = searcher.search(query, k=max_results)
        results = parse_response(hits, "it", "clef")
        save_dict["relevant_docs"] = results
        save_all.append(save_dict)

    with open(os.path.join(out_results,
                           "{}_{}_{}_search_results.pkl".format(parts, year, collection_name)),
              "wb") as outfile:
        pkl.dump(save_all, outfile)


def main(parse_args):
    mono, bili = clef_data_tables()
    if parse_args.action == "format" or parse_args.action == "index":
        subcollections = parse_args.subcollections
        if subcollections is not None:
            collection_json = parse_args.collection_json.format("-".join(parse_args.subcollections))
        else:
            collection_json = parse_args.collection_json.format("ALL")
        if subcollections is None:
            subcollections = ["ALL"]
        collection_name = "{}_{}".format("it", "-".join(subcollections))
        print(parse_args.action)
        print(collection_json)
        if not os.path.exists(collection_json):
            dump_json_from_collections(parse_args.collection_pkl, collection_json, subcollections)
        if "index" == parse_args.action:
            # Indexing data
            # TODO this might have problems, currently using index_only() method
            subprocess.check_call(['src/cedr/indexes/scripts/index.sh', collection_json, collection_name])

    elif parse_args.action == "search":
        topic_path = "data/clef_data/topics"
        for current_year in os.listdir(topic_path):
            y = current_year[-4:]
            if y not in parse_args.years:
                continue
            current_year_folder = os.path.join(topic_path, current_year, "topics")
            for file in os.listdir(current_year_folder):
                if parse_args.task not in file:
                    continue
                if parse_args.task == "MONO":
                    l = file.split("-")[-2].lower()
                    if l not in parse_args.langs:
                        continue
                    print("Working on ", file)
                    subcollections = [x[1] for x in mono[l][y]['collections']]
                    single_search_test(y, l, subcollections, os.path.join(current_year_folder, file), parse_args.parts,
                                       parse_args.max_results, parse_args.out_results)
                else:
                    raise NotImplementedError
    else:
        raise ValueError


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="MONO", choices=["MONO", "BILI", "MULTI"], help="task choice")
    parser.add_argument("--action", type=str, default="search", choices=['format', 'index', 'search'])
    parser.add_argument("--parts", type=str, default="both")
    parser.add_argument('--langs', type=str, nargs="+", default="de es fr it".split(),
                        help="languages choice, e.g.,  ar de en es fr it ja pt ru sq zh")
    parser.add_argument('--years', type=str, nargs="+", default="2000 2001 2002 2003 2004 2005 2006 2007".split(),
                        help="year choice, e.g.,  2000 2001 2002 2003 2004 2005 2006 2007")

    parser.add_argument("--max_results", type=int, default=150)
    parser.add_argument("--collection-pkl", type=str, required=False)
    parser.add_argument("--collection-json", type=str, required=False)
    parser.add_argument("--out-results", type=str, default="data/ANSERINI/results_default_bm25-rm3", required=False)
    parser.add_argument("--subcollections", type=str, nargs="+", default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    # index_only()
    # exit()
    if args.out_results:
        if not os.path.exists(args.out_results):
            os.makedirs(args.out_results)
    main(args)
