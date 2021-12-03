import os
import pickle as pkl
from tqdm import tqdm

from src.data_creation.clef_fn import read_xml_by_corpus


def clean_text(s):
    # html_escape_table
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    s = s.replace("&amp;", "&")
    s = s.replace("&quot;", '"')
    s = s.replace("&apos;", "'")
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.strip()
    # s = re.sub(' +', ' ', s)
    return s


def dump_clef_collections_format(languages, ids="str"):
    lang2clef = {'english': ['GLASGOW1995', 'LATIMES1994', 'LATIMES2002'],
                 'french': ['ATS1994', 'ATS1995', 'LEMONDE1994', 'LEMONDE1995'],
                 'german': ['ALGEMEEN1994', 'ALGEMEEN1995', 'SDA1994', 'SDA1995', 'SPIEGEL1994', 'SPIEGEL1995',
                            'FRANKFURTER1994'],
                 'italian': ['AGZ1994', 'AGZ1995', 'LASTAMPA1994'],
                 'portuguese': ['FOLHA1994', 'FOLHA1995', 'PUBLICO1994', 'PUBLICO1995'],
                 'russian': ['IZVESTIA1995'],
                 'spanish': ['EFE1994', 'EFE1995']}

    for lang in languages:
        if code2lang[lang] not in lang2clef:
            continue
        print("Working with language {} ".format(lang))
        saving_storage = dict()
        # the path to original CLEF collections
        dir_ = "CLEF/DOCUMENTS"
        # where the .pkl clean collections will be saved --> to be later used in anserini_recreate_data.py
        out_text = "clef_fix-id_collection"
        for collection in lang2clef[code2lang[lang]]:
            if not os.path.exists(out_text):
                os.makedirs(out_text)

            for file in tqdm(os.listdir(os.path.join(dir_, collection)),
                             desc="Reading collection {}".format(collection)):
                if file.endswith(".xml"):
                    parsed_file = read_xml_by_corpus(os.path.join(dir_, collection, file), collection)
                    if parsed_file is not None:
                        title, item = parsed_file
                        title = clean_text(title)
                        item['text'] = [clean_text(item['text'][0])]
                        if title not in saving_storage:
                            saving_storage[title] = item
                            if ids == "list":
                                saving_storage[title]['id'] = [saving_storage[title]['id']]
                        else:
                            if saving_storage[title]['text'] != item['text']:
                                # print("duplicate but different text", title, saving_storage[title]['headline'])
                                saving_storage[title]['text'] += item['text']
                                if ids == "list":
                                    saving_storage[title]['id'] += [item['id']]

        print("Total documents in {} are {}".format(lang, len(saving_storage)))
        with open(os.path.join(out_text, f'{lang}.pkl'), "wb") as outfile:
            pkl.dump(saving_storage, outfile)


if __name__ == "__main__":
    list_languages = "es fr it de".split()
    code = "spanish french italian german".split()
    code2lang = dict()
    for i in range(len(list_languages)):
        code2lang[list_languages[i]] = code[i]

    dump_clef_collections_format(list_languages, ids="list")
