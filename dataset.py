import os
import numpy as np
import logging
import coloredlogs
import pickle
logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)


def pickle_io(path, mode='r', obj=None):
    path = os.path.join(os.getcwd(), path)
    if mode in ['rb', 'r']:
        logger.info("Loading obj from {}...".format(path))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    elif mode in ['wb', 'w']:
        logger.info("Dumping obj to {}...".format(path))
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

def save_cache(args):
    word2id = {}
    id2word = {}
    vocab_size = 0
    embed_dim = 0

    # 统计单词个数和词向量维度
    with open(args.wordvec_file, 'r') as f:
        first = True
        while (True):
            line = f.readline()
            if line:
                vocab_size += 1
            else:
                break
            if first:
                first = False
                embed_dim = len(line.rstrip().split(" ")) - 1

    pretrained_embedding = np.random.randn(vocab_size, embed_dim)
    with open(os.path.join(os.getcwd(), args.wordvec_file), 'r') as f:
        wordid = 0
        for line in f:
            line = line.rstrip().split(" ")
            word2id[line[0]] = wordid
            id2word[wordid] = line[0]
            vector = [float(x) for x in line[1:]]
            pretrained_embedding[wordid] = vector
            wordid += 1
            # if(wordid == 100):  # debug时使用
            #     break
    norm = np.power(pretrained_embedding, 2)
    norm = np.sqrt(np.sum(norm, axis=-1, keepdims=True))
    pretrained_embedding /= norm  # 用余弦相似度做两个word vector的相似度，这里让所有vec变成模=1，之后向量内积就是cosine similarity
    np.save(args.cache_embedding, pretrained_embedding) # 保存当前词向量文件的embedding和vocab，避免多次运行时重复加载浪费时间
    pickle_io(args.word2id_file, 'wb', word2id)
    pickle_io(args.id2word_file, 'wb', id2word)
    logger.info("Cache saved!")
    return pretrained_embedding, word2id, id2word


def loadwordvec(args):
    if(os.path.exists(args.cache_embedding) and os.path.exists(args.word2id_file) and os.path.exists(args.id2word_file)):
        logger.info("Loading from cache...")
        pretrained_embedding = np.load(args.cache_embedding)
        word2id = pickle_io(args.word2id_file, mode="rb")
        id2word = pickle_io(args.id2word_file, mode="rb")
    else:
        logger.info("No cache found")
        pretrained_embedding, word2id, id2word = save_cache(args)
    vocab_size = len(word2id)
    embed_dim = pretrained_embedding.shape[-1]
    logger.info("Load pretrained embedding successfully!")
    logger.info("vocab_size = {}".format(vocab_size))
    logger.info("embed_dim = {}".format(embed_dim))
    return pretrained_embedding, word2id, id2word
