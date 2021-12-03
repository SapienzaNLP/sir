import os
from typing import List
import faiss
import datasets as ds
import numpy as np
# import gzip


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["bnid"], documents["gloss"]):
        for passage in split_text(text):
            titles.append(title)
            texts.append(passage)
    print(len(titles), len(texts))
    if len(texts) > 1000:
        print()
    return {"title": titles, "text": texts}


def load_embeddings(path):
    embeddings = dict()
    with open(path) as lines:
        _, dim = next(lines).split()
        for line in lines:
            senseid, *features = line.strip().split()
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
    glosses_file = 'data/ares/sensekey2gloss.txt'
    synset_embeddings_path = 'data/ares/ares_bert_large.txt'
    output_path = 'data/ares/ares_bert_large_faiss_index_exact'
    assert os.path.isfile(glosses_file)
    print("Step 1 - Create the dataset")

    dataset = ds.load_dataset(
        "csv", data_files=[glosses_file], split="train", delimiter="\t", column_names=["bnid", "gloss"]
    )

    dataset = dataset.map(split_documents, batched=True, num_proc=4)
    embeddings, sense_embedding_dim = load_embeddings(synset_embeddings_path)
    dataset = dataset.map(
        lambda docs: {'embeddings': [embeddings[d] for d in docs['bnid']]},
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

    # exact match index
    index = faiss.IndexFlatL2(sense_embedding_dim)
    dataset.add_faiss_index("embeddings", custom_index=index)

    # And save the index
    index_path = os.path.join(output_path, "ares_l2_index.faiss")
    dataset.get_index("embeddings").save(index_path)


if __name__ == '__main__':
    main()
