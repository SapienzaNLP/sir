python -m src.rerank \
--model sense_vanilla_bert \
--bert bert-base-multilingual-cased \
--datafiles data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title-desc_as_query.tsv data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_documents.tsv \
--topics_only data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_as_query.tsv \
--qrels data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/qrels.tsv \
--run data/clef_data/zeroshot_test/it_2001_bm25-rm3_test/title_test.run \
--out_path it_2001_bm25-rm3.test.results \
--model_weights out/robust-zeroshot_title-desc_vanillambert_ares-mult_sense-aware/best_weights.p \
--device cuda \
--sense_aware \
--freeze_query_encoder \
--normalize_query \
--use_english_only