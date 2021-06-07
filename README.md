# L1-Refinement
Code for [Cross-Lingual Word Embedding Refinement by ℓ1 Norm Optimisation](https://www.aclweb.org/anthology/2021.naacl-main.214/) 

__:see_no_evil: A more detailed readme is coming soon__


## Tested environment
- python==3.7.6
- faiss==1.6.3
- scipy==1.4.1
- numpy==1.18.1
- torch==1.6.0
- fastText==0.9.2
- Intel Core i9-9900K CPU with 32GB Memory



## Example command
```python refiner.py --src_lang en --tgt_lang de --src_emb aligned/en-de/embeddings/en.vec --tgt_emb aligned/en-de/embeddings/de.vec --exp_path a/target/dir```


## About
If you like our project or find it useful, please give us a :star: and cite us
```bib
@inproceedings{L1-Refinement,
    title = "Cross-Lingual Word Embedding Refinement by $\ell_{1}$ Norm Optimisation",
    author = "Peng, Xutan  and
      Lin, Chenghua  and
      Stevenson, Mark",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.214",
    pages = "2690--2701"
}
```

> This code is based on [MUSE](https://github.com/facebookresearch/MUSE)

