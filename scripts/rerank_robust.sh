python -m src.rerank \
--model vanilla_bert \
--bert bert-base-uncased \
--datafiles data/robust/title-desc_as_queries.tsv data/robust/documents.tsv \
--topics_only data/robust/queries.tsv \
--qrels data/robust/qrels \
--run data/robust/f1.test.run \
--out_path f1.test.results \
--device cuda \
--model_weights out/robust_title-desc_vanillabert-bert-base_noSA_f1/best_weights.p
