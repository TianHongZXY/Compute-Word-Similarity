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

    def forward(self, wordid):
        wordid = torch.LongTensor(wordid)
        wordid = wordid.to(self.args.device)
        wordvec = self.embedding(wordid)
        sim_score = torch.matmul(wordvec, self.embedding.weight.data.T)
        score, indice = torch.topk(sim_score, k=self.args.topk + 1, dim=1, largest=True, sorted=True)
        score = score[:, 1:] # 第一个score最大是该单词自身，跳过
        indice = indice[:, 1:] # index[i]即为与wordid[i]最接近的topk个单词的id
        return score.cpu().numpy(), indice.cpu().numpy()
