python -m src.train \
--model sense_vanilla_bert \
--bert bert-base-multilingual-cased \
--datafiles data/robust/title-desc_as_queries.tsv data/robust/documents.tsv \
--topics_only data/robust/queries.tsv \
--qrels data/robust/qrels \
--train_pairs data/robust/f1.train_test_merged.pairs \
--valid_run data/robust/f1.valid.run \
--device cuda \
--model_out_dir out/robust-zeroshot_title-desc_vanillambert_ares-mult \
--sense_aware \
--freeze_query_encoder \
--normalize_query \
--use_english_only
