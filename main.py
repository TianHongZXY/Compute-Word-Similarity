import time
import torch
import argparse
from calc_sim import similarity_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_word', default="", type=str, help="Input the word you want to find similar words")
    parser.add_argument('--topk', default=100, type=int, help="How many synonyms you want")
    parser.add_argument('--word2id_file', default="wordvec_files/word2id.pkl", type=str, help="cache vocab")
    parser.add_argument('--id2word_file', default="wordvec_files/id2word.pkl", type=str, help="cache vocab")
    parser.add_argument('--wordvec_file', default="wordvec_files/glove.6B.200d.txt", type=str, help="GloVe file")
    parser.add_argument('--cache_embedding', default="wordvec_files/cache.npy", type=str, help="cache embedding")

    args = parser.parse_known_args()[0]
    model = similarity_model(args)
    words = ['mother', 'father', 'eat', 'drink', 'school', 'student', 'teacher']
    with torch.no_grad():
        start_time = time.time()
        if args.pred_word:
            words = args.pred_word.rstrip().lstrip().split(" ")
        for word in words:
            model(word)
            print("-" * 50)
        print("Average time for calculating one word is {}".format((time.time() - start_time) / len(words)))