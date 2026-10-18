# Release assets: THAW measured data (data-v1)

The repository holds the code, the 77 run directories and the query dumps.
The 38 GB of measured data — encoded caches, corpora, model weights and the
database dump — is attached to the **data-v1** release because GitHub caps a
repository file at 100 MB. Assets are named after the local path with `/`
replaced by `__`, so `cache/foo.npy` is the asset `cache__foo.npy`.

Files larger than 1.9 GB are split into `.partNN` pieces. Reassemble with:

```bash
# bash / Git Bash, run in the folder that will hold the restored tree
cat cache__3844646357568e4071b35b3700ad7871.npy.part* > cache/3844646357568e4071b35b3700ad7871.npy
```

Then check each restored file against the size in the table (and, for the
npz/npy files, load the header: the first dimension is the corpus size).

## Assets (172 files, 38.2 GB)

| asset | local path | size |
|---|---|---|
| `cache__2aa13438bd13a33c3e6705a785ecfcc2.npy` | `cache/2aa13438bd13a33c3e6705a785ecfcc2.npy` | 768 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part00` | `cache/3844646357568e4071b35b3700ad7871.npy` | 1900 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part01` | `cache/3844646357568e4071b35b3700ad7871.npy` | 1900 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part02` | `cache/3844646357568e4071b35b3700ad7871.npy` | 1900 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part03` | `cache/3844646357568e4071b35b3700ad7871.npy` | 1900 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part04` | `cache/3844646357568e4071b35b3700ad7871.npy` | 1900 MB |
| `cache__3844646357568e4071b35b3700ad7871.npy.part05` | `cache/3844646357568e4071b35b3700ad7871.npy` | 330 MB |
| `cache__5d103cb11869f81b2b5023901a9fc385.f16.npy` | `cache/5d103cb11869f81b2b5023901a9fc385.f16.npy` | 768 MB |
| `cache__5d103cb11869f81b2b5023901a9fc385.npy` | `cache/5d103cb11869f81b2b5023901a9fc385.npy` | 1536 MB |
| `cache__68e87210fbce6bcf9bb97f4829778322.npy` | `cache/68e87210fbce6bcf9bb97f4829778322.npy` | 89 MB |
| `cache__8256b4ed43559037dee7d45f28746869.npy` | `cache/8256b4ed43559037dee7d45f28746869.npy` | 768 MB |
| `cache__ad3f5957fc3400a29de3e268ada99b08.npy.part00` | `cache/ad3f5957fc3400a29de3e268ada99b08.npy` | 1900 MB |
| `cache__ad3f5957fc3400a29de3e268ada99b08.npy.part01` | `cache/ad3f5957fc3400a29de3e268ada99b08.npy` | 148 MB |
| `cache__c60e09474748d76ad5f3eafb61126bce.npy.part00` | `cache/c60e09474748d76ad5f3eafb61126bce.npy` | 1900 MB |
| `cache__c60e09474748d76ad5f3eafb61126bce.npy.part01` | `cache/c60e09474748d76ad5f3eafb61126bce.npy` | 148 MB |
| `step1_data__load_report.json` | `step1_data/load_report.json` | 0 MB |
| `step1_data__load_report.json.sha256` | `step1_data/load_report.json.sha256` | 0 MB |
| `step1_data__queries.jsonl` | `step1_data/queries.jsonl` | 0 MB |
| `step1_data__texts.jsonl` | `step1_data/texts.jsonl` | 189 MB |
| `step1_data__truth.npz` | `step1_data/truth.npz` | 17 MB |
| `step1_data__vectors.npz` | `step1_data/vectors.npz` | 790 MB |
| `external__db__thaw.dump` | `external/db/thaw.dump` | 1731 MB |
| `beir__msmarco__corpus.jsonl.part00` | `beir/msmarco/corpus.jsonl` | 1900 MB |
| `beir__msmarco__corpus.jsonl.part01` | `beir/msmarco/corpus.jsonl` | 1471 MB |
| `beir__msmarco__qrels__dev.tsv` | `beir/msmarco/qrels/dev.tsv` | 0 MB |
| `beir__msmarco__qrels__test.tsv` | `beir/msmarco/qrels/test.tsv` | 0 MB |
| `beir__msmarco__qrels__train.tsv` | `beir/msmarco/qrels/train.tsv` | 0 MB |
| `beir__msmarco__queries.jsonl` | `beir/msmarco/queries.jsonl` | 32 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00000-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00000-of-00157.parquet` | 545 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00001-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00001-of-00157.parquet` | 546 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00002-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00002-of-00157.parquet` | 546 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00003-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00003-of-00157.parquet` | 546 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00004-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00004-of-00157.parquet` | 546 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00005-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00005-of-00157.parquet` | 545 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00006-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00006-of-00157.parquet` | 544 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00007-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00007-of-00157.parquet` | 537 MB |
| `external__wiki_dpr_clean__data__psgs_w100__nq__train-00008-of-00157.parquet` | `external/wiki_dpr_clean/data/psgs_w100/nq/train-00008-of-00157.parquet` | 530 MB |
| `external__models__huggingface__hub__datasets--BeIR--msmarco__refs__main` | `external/models/huggingface/hub/datasets--BeIR--msmarco/refs/main` | 0 MB |
| `external__models__huggingface__hub__datasets--BeIR--msmarco__snapshots__a918e0d11a77ed33f42f29d98340b655593b96ad__README.md` | `external/models/huggingface/hub/datasets--BeIR--msmarco/snapshots/a918e0d11a77ed33f42f29d98340b655593b96ad/README.md` | 0 MB |
| `external__models__huggingface__hub__datasets--facebook--wiki_dpr__refs__main` | `external/models/huggingface/hub/datasets--facebook--wiki_dpr/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__refs__main` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__1_Pooling__config.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/1_Pooling/config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__config.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__config_sentence_transformers.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/config_sentence_transformers.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__model.safetensors` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/model.safetensors` | 438 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__modules.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/modules.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__README.md` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__sentence_bert_config.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/sentence_bert_config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__special_tokens_map.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__tokenizer.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer.json` | 1 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__tokenizer_config.json` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-base-en-v1.5__snapshots__a5beb1e3e68b9ab74eb54cfd186867f64f240e1a__vocab.txt` | `external/models/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-m3__refs__main` | `external/models/huggingface/hub/models--BAAI--bge-m3/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__refs__main` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__.gitattributes` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/.gitattributes` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__1_Pooling__config.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/1_Pooling/config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__config.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__config_sentence_transformers.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config_sentence_transformers.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__model.safetensors` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors` | 133 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__modules.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/modules.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__onnx__model.onnx` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx` | 133 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__pytorch_model.bin` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/pytorch_model.bin` | 134 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__README.md` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__sentence_bert_config.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/sentence_bert_config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__special_tokens_map.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__tokenizer.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json` | 1 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__tokenizer_config.json` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--BAAI--bge-small-en-v1.5__snapshots__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a__vocab.txt` | `external/models/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__refs__main` | `external/models/huggingface/hub/models--bert-base-uncased/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__snapshots__86b5e0934494bd15c9632b12f734a8a67f723594__config.json` | `external/models/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/config.json` | 0 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__snapshots__86b5e0934494bd15c9632b12f734a8a67f723594__model.safetensors` | `external/models/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/model.safetensors` | 440 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__snapshots__86b5e0934494bd15c9632b12f734a8a67f723594__tokenizer.json` | `external/models/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json` | 0 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__snapshots__86b5e0934494bd15c9632b12f734a8a67f723594__tokenizer_config.json` | `external/models/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--bert-base-uncased__snapshots__86b5e0934494bd15c9632b12f734a8a67f723594__vocab.txt` | `external/models/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__refs__main` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__refs__refs__pr__3` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/refs/refs/pr/3` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__b5380dc2f0ca210dcf269f9c9d3d2259347d5e2b__model.safetensors` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/b5380dc2f0ca210dcf269f9c9d3d2259347d5e2b/model.safetensors` | 438 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__bb21a3c2b1656d60c6a8e920283bc40dabddadb8__config.json` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/bb21a3c2b1656d60c6a8e920283bc40dabddadb8/config.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__bb21a3c2b1656d60c6a8e920283bc40dabddadb8__pytorch_model.bin` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/bb21a3c2b1656d60c6a8e920283bc40dabddadb8/pytorch_model.bin` | 438 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__bb21a3c2b1656d60c6a8e920283bc40dabddadb8__tokenizer.json` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/bb21a3c2b1656d60c6a8e920283bc40dabddadb8/tokenizer.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__bb21a3c2b1656d60c6a8e920283bc40dabddadb8__tokenizer_config.json` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/bb21a3c2b1656d60c6a8e920283bc40dabddadb8/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-ctx_encoder-single-nq-base__snapshots__bb21a3c2b1656d60c6a8e920283bc40dabddadb8__vocab.txt` | `external/models/huggingface/hub/models--facebook--dpr-ctx_encoder-single-nq-base/snapshots/bb21a3c2b1656d60c6a8e920283bc40dabddadb8/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__config.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/config.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__refs__main` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__config.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/config.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__pytorch_model.bin` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/pytorch_model.bin` | 438 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__special_tokens_map.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__tokenizer.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/tokenizer.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__tokenizer_config.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__snapshots__d04a52f6d2f96c60117a925e8c24c4043a75f265__vocab.txt` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/snapshots/d04a52f6d2f96c60117a925e8c24c4043a75f265/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--facebook--dpr-question_encoder-single-nq-base__trees__d04a52f6d2f96c60117a925e8c24c4043a75f265.json` | `external/models/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/trees/d04a52f6d2f96c60117a925e8c24c4043a75f265.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__refs__main` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__1_Pooling__config.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/1_Pooling/config.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__config.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/config.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__model.safetensors` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/model.safetensors` | 438 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__modules.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/modules.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__README.md` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/README.md` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__sentence_bert_config.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/sentence_bert_config.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__special_tokens_map.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__tokenizer.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/tokenizer.json` | 1 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__tokenizer_config.json` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--intfloat--e5-base-v2__snapshots__f52bf8ec8c7124536f0efb74aca902b2995e5bcd__vocab.txt` | `external/models/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--Qwen--Qwen3-Embedding-0.6B__refs__main` | `external/models/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__refs__main` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__1_Pooling__config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__config_sentence_transformers.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config_sentence_transformers.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__model.safetensors` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors` | 91 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__modules.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/modules.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__README.md` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__sentence_bert_config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__special_tokens_map.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__tokenizer.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__tokenizer_config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-MiniLM-L6-v2__snapshots__1110a243fdf4706b3f48f1d95db1a4f5529b4d41__vocab.txt` | `external/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/vocab.txt` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__refs__main` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/refs/main` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__1_Pooling__config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/1_Pooling/config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__config_sentence_transformers.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/config_sentence_transformers.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__model.safetensors` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/model.safetensors` | 438 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__modules.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/modules.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__README.md` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__sentence_bert_config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/sentence_bert_config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__special_tokens_map.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/special_tokens_map.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__tokenizer.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/tokenizer.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__tokenizer_config.json` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/tokenizer_config.json` | 0 MB |
| `external__models__huggingface__hub__models--sentence-transformers--all-mpnet-base-v2__snapshots__e8c3b32edf5434bc2275fc9bab85f82640a19130__vocab.txt` | `external/models/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/vocab.txt` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__.gitattributes` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/.gitattributes` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__1_Pooling__config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/1_Pooling/config.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__colbert_linear.pt` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/colbert_linear.pt` | 2 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/config.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__config_sentence_transformers.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/config_sentence_transformers.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__configuration.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/configuration.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__bm25.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/bm25.jpg` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__long.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/long.jpg` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__miracl.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/miracl.jpg` | 1 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__mkqa.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/mkqa.jpg` | 1 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__nqa.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/nqa.jpg` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__imgs__others.webp` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/imgs/others.webp` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__long.jpg` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/long.jpg` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__modules.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/modules.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/config.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__Constant_7_attr__value` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/Constant_7_attr__value` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__model.onnx` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/model.onnx` | 1 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__model.onnx_data.part00` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/model.onnx_data` | 1900 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__model.onnx_data.part01` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/model.onnx_data` | 367 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__sentencepiece.bpe.model` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/sentencepiece.bpe.model` | 5 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__special_tokens_map.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/special_tokens_map.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__tokenizer.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/tokenizer.json` | 17 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__onnx__tokenizer_config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/onnx/tokenizer_config.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__pytorch_model.bin.part00` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/pytorch_model.bin` | 1900 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__pytorch_model.bin.part01` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/pytorch_model.bin` | 371 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__README.md` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/README.md` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__sentence_bert_config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/sentence_bert_config.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__sentencepiece.bpe.model` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/sentencepiece.bpe.model` | 5 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__sparse_linear.pt` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/sparse_linear.pt` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__special_tokens_map.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/special_tokens_map.json` | 0 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__tokenizer.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/tokenizer.json` | 17 MB |
| `external__models__modelscope__models__BAAI--bge-m3__snapshots__master__tokenizer_config.json` | `external/models/modelscope/models/BAAI--bge-m3/snapshots/master/tokenizer_config.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__.gitattributes` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/.gitattributes` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__1_Pooling__config.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/1_Pooling/config.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__config.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/config.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__config_sentence_transformers.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/config_sentence_transformers.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__configuration.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/configuration.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__generation_config.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/generation_config.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__merges.txt` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/merges.txt` | 2 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__model.safetensors` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/model.safetensors` | 1192 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__modules.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/modules.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__README.md` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/README.md` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__tokenizer.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/tokenizer.json` | 11 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__tokenizer_config.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/tokenizer_config.json` | 0 MB |
| `external__models__modelscope__models__Qwen--Qwen3-Embedding-0.6B__snapshots__master__vocab.json` | `external/models/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master/vocab.json` | 3 MB |

## Restoring the database

```bash
pg_restore -d thaw external/db/thaw.dump
```

## What each piece is

- `cache/` — the encoded matrices keyed by content hash: the 500k spaces for
  MiniLM, bge-small, bge-base, bge-m3 and Qwen3-Embedding-0.6B, the 8.8M
  MiniLM space, and the f16 variants. This is the expensive part (hours of
  GPU) and the only data here that had to exist on a particular machine.
- `step1_data/` — the step1 vectors and the queries/texts they were built from.
- `external/db/thaw.dump` — the loaded PostgreSQL database: 500k documents
  with both models' vectors, the table `thaw_pg` reads.
- `beir/` — the MS MARCO corpus in BEIR layout (public data, included for
  convenience; `--corpus beir --path beir/` is the form the runs use).
- `external/wiki_dpr_clean/` — the wiki_dpr parquet shards the DPR estimates
  read (public data from facebook/wiki_dpr).
- `external/models/` — the encoder weights (public, from huggingface and
  modelscope): MiniLM, bge-small/base/m3, Qwen3-Embedding-0.6B, the DPR
  question and context encoders, e5 and mpnet.

One matrix is missing and cannot be restored: the 8.8M bge-base f16 file
existed only on a cloud instance that has since been released. It gates local
re-runs of the scaling-mechanism analysis; the numbers it produced are in
`results/runs/`.
