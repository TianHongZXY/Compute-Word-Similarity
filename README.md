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