#!/bin/sh
collection=$1
collection_name=$2
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input $collection \
 -index data/clef_data/clef_anserini_indexes/$collection_name \
 -storePositions -storeDocvectors -storeContents