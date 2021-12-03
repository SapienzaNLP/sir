import os
from typing import List
import faiss
import datasets as ds
import numpy as np
# import gzip


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def process_text(bnid, text, collection, bnid_collection):
    if bnid == 'bnid':
        return
    for passage in split_text(text):
        collection.append(passage)
        bnid_collection.append(bnid)


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    bnids, en_texts, it_texts, es_texts, fr_texts, de_texts = [], [], [], [], [], []
    for title, en, it, es, fr, de in zip(documents["bnid"], documents["EN"], documents["IT"], documents["ES"],
                                         documents["FR"], documents["DE"]):
        # if title == 'bnid': continue
        bnids.append(title)
        en_texts.append(en)
        it_texts.append(it)
        es_texts.append(es)
        fr_texts.append(fr)
        de_texts.append(de)

    return {"bnid": bnids, "title": [""] * len(en_texts), "text": [""] * len(en_texts), "EN": en_texts, 'IT': it_texts,
            'ES': es_texts, 'FR': fr_texts, 'DE': de_texts}


def load_embeddings(path, stopsynsets):
    embeddings = dict()
    with open(path) as lines:
        _, dim = next(lines).split()
        for line in lines:
            senseid, *features = line.strip().split()
            if senseid in stopsynsets:
                continue
            embedding = np.array([float(x) for x in features])
            embedding = embedding / np.linalg.norm(embedding, 2, 0)
            embeddings[senseid] = embedding
    return embeddings, int(dim)


def main():
    # The dataset needed for RAG must have three columns:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # - embeddings (array of dimension d): DPR representation of the passage

    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    glosses_file = 'data/ares/EN-DE-FR-ES-IT.glosses.txt'  # 'data/in/bn_offset_to_gloss.txt'
    synset_embeddings_path = 'data/ares/ares_bert_base_multilingual.ugly_fix.txt'
    output_path = 'data/ares/normalized_faiss_index_no_stopsynsets_multilingual_exact_match'
    stop_synsets_path = 'data/ares/stop_synsets.txt'
    with open(stop_synsets_path) as lines:
        stopsynsets = set(line.strip() for line in lines)

    assert os.path.isfile(glosses_file)
    print("Step 1 - Create the dataset")

    dataset = ds.load_dataset(
        "csv", data_files=[glosses_file], split="train", delimiter="\t",
        column_names=["bnid", "EN", 'IT', 'ES', 'FR', 'DE'])

    dataset = dataset.filter(lambda doc: doc['bnid'] not in stopsynsets and doc['bnid'] != 'bnid').map(split_documents,
                                                                                                       batched=True,
                                                                                                       num_proc=4)
    embeddings, sense_embedding_dim = load_embeddings(synset_embeddings_path, stopsynsets)
    dataset = dataset.map(lambda docs: {'embeddings': [embeddings[d] for d in docs['bnid']]},
                          batched=True,
                          batch_size=32,
                          )

    # And finally save your dataset
    dataset.save_to_disk(output_path)
    # from datasets import load_from_disk
    # dataset = load_from_disk(passages_path)  # to reload the dataset

    ######################################
    print("Step 2 - Index the dataset")
    ######################################

    # index = faiss.IndexFlatIP(sense_embedding_dim)
    index = faiss.IndexFlatL2(sense_embedding_dim)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(output_path, "ares_hnsw_index.faiss")
    dataset.get_index("embeddings").save(index_path)


if __name__ == '__main__':
    main()
