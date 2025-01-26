import os

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


path_dir = os.path.join(os.getcwd(), 'models/embedding_model')

glove_file = datapath(os.path.join(path_dir, 'glove.42B.300d.txt'))
tmp_file = get_tmpfile(os.path.join(path_dir, 'test.txt'))
_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
model.save(os.path.join(path_dir, 'glove.42B.300d.d2v'))