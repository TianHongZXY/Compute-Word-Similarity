import time
from tqdm import tqdm
import torch
import argparse
from calc_sim import similarity_model
from dataset import pickle_io

def print_result(wordids, id2word, score, indice):
    for i in range(score.shape[0]):
        print(id2word[wordids[i]])
        for s, inx in zip(score[i], indice[i]):
            print("({}, {})".format(s, id2word[inx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_word', default="", type=str, help="Input the word you want to find similar words")
    parser.add_argument('--topk', default=100, type=int, help="How many synonyms you want")
    parser.add_argument('--word2id_file', default="wordvec_files/word2id.pkl", type=str, help="Cache vocab")
    parser.add_argument('--id2word_file', default="wordvec_files/id2word.pkl", type=str, help="Cache vocab")
    parser.add_argument('--wordvec_file', default="wordvec_files/glove.6B.200d.txt", type=str, help="GloVe file")
    parser.add_argument('--cache_embedding', default="wordvec_files/cache.npy", type=str, help="Cache embedding")
    parser.add_argument('--use_gpu', default=False, type=bool, help="Enable gpu")
    parser.add_argument('--device', default='cpu', type=str, help="torch device")
    parser.add_argument('--batch_size', default=1000, type=int, help="batch size")
    parser.add_argument('--print_result', default=False, type=bool, help="If print topk words")

    args = parser.parse_known_args()[0]

    model = similarity_model(args)
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()
        args.device = 'cuda'
    id2word = pickle_io(args.id2word_file, mode="rb")
    wordids = list(id2word.keys())
    with torch.no_grad():
        start_time = time.time()
        index = 0
        num_batch = len(wordids) // args.batch_size # 完整batch数目
        # while(index + args.batch_size <= len(wordids)):
        for index in tqdm(range(num_batch)):
            score, indice = model(wordids[index * args.batch_size:(index + 1) * args.batch_size])
            if args.print_result:
                print_result(wordids, id2word, score, indice)
        score, indice = model(wordids[num_batch * args.batch_size:]) # 最后一个不足batch_size的batch
        if args.print_result:
            print_result(wordids, id2word, score, indice)
            # print("-" * 50)
        # print("Average time for calculating one word is {}".format((time.time() - start_time) / len(words)))