# Dialog dataset for compositional semantic parsing

This repository contains the dialog dataset for evaluating compositional generalization of semantic parsing models, in the paper
[Grammar-based Decoding for Improved Compositional Generalization in Semantic Parsing](https://aclanthology.org/2023.findings-acl.91).

The dialog corpus has been converted into samples, where each sample represents a dialog turn, with the following information:
- id
- text: the utterance of the current turn
- extra: the utterance of the previous turn
- target: the function expression of the current turn

To evaluation compositional generalization, we provide the following splits:
1. **simple**: a split where train/test have similar data distribution.
2. **length**: a split based on the max number of arguments in the target expression
3. 6 **tmcd** splits: base on the MCD principle, these splits exhibit different degree of compound divergence,
   while their atom divergence is kept the same at 0.01.

Each split contains ``{train,dev,test}.jsonl``.
Note that the dev set shares the same distribution as the test set, and thus should be used strictly only for validation, not for training.

## Baseline ONMT experiments

The following describes how to run the onmt baseline experiments.

For each split, the exact random seeds (55 of them) used in the onmt baseline experiments are recorded in the file ``seeds_info.txt``.

1. First, get the Chinese wordvec, ``cc.zh.300.vec``, from https://fasttext.cc/docs/en/crawl-vectors.html.

   ```bash
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
   gunzip cc.zh.300.vec.gz
   ```  

2. Create env and install python packages

   ```bash
   virtualenv --python=python3.7.11 env
   source env/bin/activate

   pip install openNMT-py==1.2.0 torch==1.12.1+cu113
   ```  

3. Use the run.sh script to run traing, testing and get accuracy.
   Change the output_root, input_root and wordvector to the right place.
   Then, run for a given dataset, gpu_id, and seed:  

   ```bash
   'sh ./run.sh {dataset_dir} {gpu_id} {seed}
   ```  
   for example, `nohup sh ./run.sh tmcd.12088598 0 1 > 1.log 2>&1 &`.  
   The last line in the log shows the accuracy:
   ```
   accuracy = 0.6651 (21057/31662)
   ```
   We reported the average of the 55 runs with different seeds for each split (see seeds_info.txt).

## Citation

```bibtex
@inproceedings{zheng-etal-2023-grammar,
  title = "Grammar-based Decoding for Improved Compositional Generalization in Semantic Parsing",
  author = "Jing Zheng and Jyh-Herng Chow and Zhongnan Shen and Peng Xu",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
  month = jul,
  year = "2023",
  address = "Toronto, Canada",      
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.findings-acl.91",
  pages = "1399--1418",
}
```

