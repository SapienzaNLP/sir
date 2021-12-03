python -m src.train \
--model vanilla_bert \
--bert bert-base-uncased \
--datafiles data/robust/title-desc_as_queries.tsv data/robust/documents.tsv \
--topics_only data/robust/queries.tsv \
--qrels data/robust/qrels \
--train_pairs data/robust/f1.train.pairs \
--valid_run data/robust/f1.valid.run \
--device cuda \
--model_out_dir out/robust_title-desc_vanillabert-bert-base_noSA_f1 \
