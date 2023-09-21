# TernaryCL
## Alleviating Sparsity of Open Knowledge Graphs with Ternary Contrastive Learning

Source code for our EMNLP-2022-Findings paper: Alleviating Sparsity of Open Knowledge Graphs with Ternary Contrastive Learning

### Dependencies:

* Compatible with Pytorch 1.8 and Python 3.8.

### Dataset:
* ReVerb45k, ReVerb20K are included with the repository present in the `Data` directory.
* Both datasets contain the following files:

  ```shell
  ent2id.txt: all noun phrases and corresponding ids, one per line. The first line is the number of noun phrases.

  rel2id.txt: all relation phrases and corresponding ids, one per line. The first line is the number of relations.

  train_trip.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(s,r,o)*** which indicates there is a relation ***rel*** between ***s*** and ***o*** .
  
  test_trip.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(s,r,o)*** .

  valid_trip.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(s,r,o)*** .
  
  gold_npclust.txt: The ground truth noun phrase canonicalization information. This information is used during evaluations. Each line corresponds to the canonicalization information of a noun phrase in the following format ***(NP_id, no. of canonical NPs, list ids of canonical NPs)*** .
  ```
  
### Code:

* Pretrain:
  ```shell
  cd code/TernaryCL
  python main.py -dataset ReVerb20K -lr 0.00005 -neg_samples 50 -rel_neg_samples 10 -n_epochs 500 -early_stop 10
  ```
* Finetune:
  ```shell
  cd code/TernaryCL_Finetune
  python main.py -dataset ReVerb20K -lr 0.00005 -n_epochs 500 -early_stop 50
  ```

### Citation
Please cite our paper if you use this code in your work.

        @article{li2022alleviating, 
        title={Alleviating Sparsity of Open Knowledge Graphs with Ternary Contrastive Learning}, 
        author={Li, Qian and Joty, Shafiq and Wang, Daling and Feng, Shi and Zhang, Yifei}, 
        journal={Findings of EMNLP 2022}, 
        year={2022} }


For any clarification, comments, or suggestions please create an issue or contact feiwangyuzhou@foxmail.com
