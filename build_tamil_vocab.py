import os
import json
from pprint import pprint
from transformers import AutoTokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

DATA_DIR = os.environ['DATA_DIR']

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()

files = [os.environ['CORPUS_PATH']]

tokenizer.train(files, trainer)
output = tokenizer.save("{}/tokenizer.json".format(DATA_DIR), pretty=True)

pprint(output)

with open('{}/vocab.txt'.format(DATA_DIR), 'w') as f:
    tokens = [ i[0] for i in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1] )]
    f.write('\n'.join(tokens)+'\n')
