## Dialog dataset for compositional semantic parsing

This repository contains a dialog dataset for evaluating compositional generalization of semantic parsing models.

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

