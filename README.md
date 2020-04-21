# Word Similarity Calculation

## Dependencies

```bash
virtualenv -p python3 <your_virtualenv_path>
source <your_virtualenv_path>/bin/activate
pip3 install -r requirements.txt
```

## GloVe

You can download pretrained word embeddings like `glove.6B.200d.txt` from [here](https://nlp.stanford.edu/projects/glove/)

## Use from scratch

After download the glove files, put the `glove.txt` file under directory `wordvec_files/`, then to create pretrained embedding and vocab cache using the glove file, you should run `python main.py` in the terminal, it takes about 30 seconds on my MacBook Pro 2018(13.3-inch) and it will print a few examples as below:

![](https://github.com/TianHongZXY/Compute-Word-Similarity/tree/master/images/img1.png)

The you can play with it with a much faster speed, for instance:

`python main.py --pred_word love --topk 10` means find the most similar 10 words of `love` according to the cosine similarity of their word vectors.

![](https://github.com/TianHongZXY/Compute-Word-Similarity/tree/master/images/img2.png)

