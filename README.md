# Word Similarity Calculation

## Dependencies

```bash
virtualenv -p python3 <your_virtualenv_path>
source <your_virtualenv_path>/bin/activate
pip3 install -r requirements.txt
```

## GloVe

The `glove.6B.200d.txt` under `wordvec_files/` contains only the first 300 words from the original file. You can download the original files like `glove.6B.200d.txt` from [here](https://nlp.stanford.edu/projects/glove/)

## Use from scratch

After download the glove files, put the `glove.6B.200d.txt` file under directory `wordvec_files/`, if you do not use the `glove.6B.200d.txt`, you need to change the args of wordvec_file in `main.py`, then to create pretrained embedding and vocab cache using the glove file, you should run `python main.py` in the terminal, it takes about 30 seconds on my MacBook Pro 2018(13.3-inch) and it will print a few examples as below:

![](https://github.com/TianHongZXY/Compute-Word-Similarity/blob/master/images/img1.png)

The you can play with it with a much faster speed, for instance:

`python main.py --pred_word love --topk 10` means find the most similar 10 words of `love` according to the cosine similarity between their word vectors.

![](https://github.com/TianHongZXY/Compute-Word-Similarity/blob/master/images/img2.png)

For GPU user, you can specify `--use_gpu True` in terminal only if `torch.cuda.is_available() == True`, otherwise the model will compute on  cpu as default.

## Compare with Gensim

*Test on my MacBook Pro 2018(13.3-inch)*

| Metrics | My implementation | Gensim |
| :----- | :----- | :----- |
| Time to load data | 40+  seconds first time and less than 1 second after that | 120  seconds |
| Time to find 100 similar words | 0.65 seconds | 0.22 seconds |
| Time to find 1000 similar words | 0.76 seconds | 0.15 seconds |
| Time to find 10000 similar words | 0.91 seconds | 0.04 seconds |

My implementation costs more time when the `topk` becomes bigger because I use a heap to keep the topk most similar words rather than sort the cosine similarity between the target word and the whole vocabs and then return the first topk words as `Gensim` does, which is much faster than my impletation when topk is very big.

But due to the time to load `wordvec_file` for `Gensim` is too long——**up to about 2 minutes**, which is very annoying when you want to test a few different words many times. My implementation takes **less than 1 second** to load `wordvec_file` which is much faster, so it is very easy to play with it repeatly without waiting for a long time everytime.