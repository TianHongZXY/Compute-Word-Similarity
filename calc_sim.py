import torch
import torch.nn as nn
from dataset import loadwordvec
import heapq

class similarity_model(nn.Module):
    def __init__(self, args):
        super(similarity_model, self).__init__()
        self.topk = args.topk
        self.args = args
        pretrained_embedding, self.word2id, self.id2word = loadwordvec(args)
        self.vocab_size = len(self.word2id)
        self.embed_dim = pretrained_embedding.shape[-1]
        self.embedding = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        assert self.topk < self.vocab_size

    def forward(self, word):
        self.pq = []
        wordid = torch.LongTensor([self.word2id[word]])
        wordid = wordid.to(self.args.device)
        wordvec = self.embedding(wordid)
        sim_score = torch.matmul(wordvec, self.embedding.weight.data.T)
        sim_score = sim_score.cpu().numpy().tolist()[0]
        wordids = list(range(self.vocab_size))
        # zip_of_score_i = zip(sim_score, wordids)
        # zip_of_score_id = sorted(zip_of_score_id,reverse=True, key=lambda x: x[0]) # 根据sim_score对wordids排序，全词表排序，耗时长
        # sim_score, wordids = zip(*zip_of_score_id)

        for score, wordid in zip(sim_score, wordids): # 维持一个小根堆保存前topk + 1大的word
            if(len(self.pq) < self.topk + 1):
                heapq.heappush(self.pq, (score, wordid))
            else:
                smallest = heapq.heappop(self.pq)
                heapq.heappush(self.pq, (score, wordid)) if score > smallest[0] else heapq.heappush(self.pq, smallest)

        print("The top {} similar words of '{}' are: ".format(self.topk, word))
        self.pq.sort(key=lambda x:x[0], reverse=True)
        for (score, wordid) in self.pq[1:]: # 第一个是该单词自身，跳过
            print("({}, {})".format(self.id2word[wordid], score))
