import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as Et

'''ENGLISH CORPORA'''


def read_xml_glas(path):
    item = dict()
    tree = Et.parse(path)
    root = tree.getroot()
    title = root.find("HEADLINE").text.strip()
    if title == "No Headline Present":
        return None
    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [root.find("TEXT").text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = root.find("HEADLINE").text.strip()
    if root.find('BYLINE') is not None:
        item['misc'] = root.find('BYLINE').text
    else:
        item['misc'] = ""

    return title, item


def read_xml_la94(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    headline = list()
    for el in root.find("HEADLINE").find_all("P"):
        headline.append(el.text.strip())
    title = " ".join(headline)
    if title == "No Headline Present":
        return None
    sents = list()
    for el in root.find("TEXT").find_all("P"):
        sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


def read_xml_la02(path):
    item = dict()
    tree = Et.parse(path)
    root = tree.getroot()
    title = ""
    if root.find("HD").text is not None:
        title = root.find("HD").text.strip()
    if title == "No Headline Present":
        return None
    sents = list()
    for el in root.findall("LD"):
        if el.text is not None:
            sents.append(el.text.strip())
    for el in root.findall("TE"):
        if el.text is not None:
            sents.append(el.text.strip())
    if len(sents) == 0:
        return None
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


'''FRENCH CORPORA'''


def read_xml_ats94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')
    title = root.find("TI").text.strip().replace("\n", " ")
    if title == "No Headline Present":
        return None
    sents = list()
    if root.find_all("LD") is not None:
        for el in root.find_all("LD"):
            if el.text is not None:
                sents.append(el.text.strip())
    if root.find_all("TX") is not None:
        for el in root.find_all("TX"):
            if el.text is not None:
                sents.append(el.text.strip())
    if len(sents) == 0:
        return None
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


def read_xml_lemonde94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    title = root.find("TITLE").text.strip().replace("\n", " ")
    if title == "No Headline Present":
        return None
    sents = list()
    for el in root.find_all("TEXT"):
        sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    if root.find("SUBJECTS") is not None:
        item['misc'] = root.find("SUBJECTS").text.strip()
    else:
        item['misc'] = ""

    return title, item


'''ITALIAN CORPORA'''


def read_xml_agz94(path):
    return read_xml_ats94_95(path)


def read_xml_agz95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')
    all_body = root.find("TI").text.strip()
    sents = all_body.split("\n")
    if len(sents) < 3:
        return None
    title = sents[0]
    if title == "No Headline Present":
        return None
    text = " ".join(sents[2:])

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


def read_xml_lastampa94(path):
    return read_xml_lemonde94_95(path)


'''SPANISH CORPORA'''


def read_xml_efe94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    title = root.find("TITLE").text.strip().replace("\n", " ")
    if title == "No Headline Present":
        return None
    sents = list()
    for el in root.find_all("TEXT"):
        sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    if root.find("CATEGORY") is not None:
        item['misc'] = root.find("CATEGORY").text.strip()
    else:
        item['misc'] = ""

    return title, item


'''RUSSIAN CORPORA'''


def read_xml_iz95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    title = root.find("TITLE").text.strip().replace("\n", " ")
    if title == "No Headline Present":
        return None
    sents = list()
    if root.find("RETRO") is not None:
        sents.append(root.find("RETRO").text.strip())
    for el in root.find_all("TEXT"):
        sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    if root.find("SUBJECT") is not None:
        item['misc'] = root.find("SUBJECT").text.strip()
    else:
        item['misc'] = ""

    return title, item


'''GERMAN CORPORA'''


def read_xml_sda94_95(path):
    return read_xml_ats94_95(path)


def read_xml_alg94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    headline = list()
    for el in root.find("TI").find_all("P"):
        headline.append(el.text.strip())
    title = " ".join(headline)
    if title == "No Headline Present":
        return None
    sents = list()
    if root.find("LE") is not None:
        for el in root.find("LE").find_all("P"):
            sents.append(el.text.strip())
    if root.find("TE") is not None:
        for el in root.find("TE").find_all("P"):
            sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


def read_xml_frank94(path):
    return read_xml_lemonde94_95(path)


def read_xml_spiegel94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')

    title = root.find("TITLE").text.strip().replace("\n", " ")
    if title == "No Headline Present":
        return None
    sents = list()
    for el in root.find_all("LEAD"):
        sents.append(el.text.strip())
    for el in root.find_all("TEXT"):
        sents.append(el.text.strip())
    text = " ".join(sents)

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


'''POTOGUESE CORPORA'''


def read_xml_fohla94_95(path):
    item = dict()
    infile = open(path, "r")
    contents = infile.read()
    root = BeautifulSoup(contents, 'xml')
    all_body = root.find("TEXT").text.strip()
    sents = all_body.split("\n")
    if len(sents) < 2:
        return None
    title = sents[0]
    if title == "No Headline Present":
        return None
    text = " ".join(sents[1:])

    item['title'] = title
    item['id'] = root.find("DOCNO").text.strip()
    item['url'] = ''
    item['text'] = [text.replace("\n", " ")]
    item['hyperlinks'] = list()
    item['headline'] = title
    item['misc'] = ""

    return title, item


'''GENERAL'''


def read_xml_by_corpus(path, corpus):
    article = None
    try:
        if corpus == "GLASGOW1995":
            article = read_xml_glas(path)
        elif corpus == "LATIMES1994":
            article = read_xml_la94(path)
        elif corpus == "LATIMES2002":
            article = read_xml_la02(path)
        elif corpus in ["LEMONDE1994", "LEMONDE1995"]:
            article = read_xml_lemonde94_95(path)
        elif corpus in ["ATS1994", "ATS1995"]:
            article = read_xml_ats94_95(path)
        elif corpus == "AGZ1994":
            article = read_xml_agz94(path)
        elif corpus == "AGZ1995":
            article = read_xml_agz95(path)
        elif corpus == "LASTAMPA1994":
            article = read_xml_lastampa94(path)
        elif corpus in ["EFE1994", "EFE1995"]:  # index_only()
    # exit()
            article = read_xml_efe94_95(path)
        elif corpus == "IZVESTIA1995":
            article = read_xml_iz95(path)
        elif corpus in ['SDA1994', 'SDA1995']:
            article = read_xml_sda94_95(path)
        elif corpus in ['ALGEMEEN1994', 'ALGEMEEN1995']:
            article = read_xml_alg94_95(path)
        elif corpus in ['SPIEGEL1994', 'SPIEGEL1995']:
            article = read_xml_spiegel94_95(path)
        elif corpus == "FRANKFURTER1994":
            article = read_xml_frank94(path)

    except Exception as e:
        # pass
        print("Error encountered {} - {}".format(path, e))

    return article
