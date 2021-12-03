python -m src.rerank \
--model vanilla_bert \
--bert bert-base-multilingual-cased \
--datafiles data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title-desc_as_query.tsv data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_documents.tsv \
--topics_only data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_as_query.tsv \
--qrels data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/qrels.tsv \
--run data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_test.run \
--out_path it_2001_bm25-rm3.test.results \
--model_weights out/robust-zeroshot_title-desc_vanillambert/weights0.p \
--device cuda \


