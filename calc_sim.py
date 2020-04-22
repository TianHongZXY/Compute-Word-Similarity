import torch
import torch.nn as nn
from dataset import loadwordvec


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

    def forward(self, word):
        wordid = torch.LongTensor([self.word2id[word]])
        wordid = wordid.to(self.args.device)
        wordvec = self.embedding(wordid)
        sim_score = torch.matmul(wordvec, self.embedding.weight.data.T)
        sim_score = sim_score.detach().numpy().tolist()[0]
        wordids = list(range(self.vocab_size))
        zip_of_score_id = zip(sim_score, wordids)
        zip_of_score_id = sorted(zip_of_score_id,reverse=True, key=lambda x: x[0]) # 根据sim_score对wordids排序
        sim_score, wordids = zip(*zip_of_score_id)
        assert self.topk < self.vocab_size
        print("The top {} similar words of '{}' are: ".format(self.topk, word))
        for i in range(1, self.topk + 1): # 第一个是该单词自身，跳过
            print("({}, {})".format(self.id2word[wordids[i]], sim_score[i]))
