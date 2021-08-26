export ELECTRA_DIR=temp-test
export DATA_DIR=$ELECTRA_DIR/data
export TRAIN_SIZE=100000000
export MODEL_NAME=electra-tamil 
export CORPUS_URL=http://transfer.sh/1dR7TSL/tamil-corpus.txt.zip
export CORPUS_PATH=$DATA_DIR/corpus.uniq.txt

# make data directory: this is where model, pretrained records, vocab will live
mkdir -p $ELECTRA_DIR
mkdir -p $DATA_DIR

cd $ELECTRA_DIR

virtualenv -p python3 env
source env/bin/activate

pip3 install tensorflow-gpu==1.15
pip3 install transformers==2.8.0
git clone https://github.com/vanangamudi/electra-tamil.git 

#download corpus
if [ ! -f $DATA_DIR/corpus.uniq.zip ]; then
    curl $CORPUS_URL -o $DATA_DIR/corpus.uniq.zip
    unzip -o $DATA_DIR/corpus.uniq.zip -d $DATA_DIR
fi

if [ ! -f $DATA_DIR/corpus.uniq.txt ]; then
    unzip -o $DATA_DIR/corpus.uniq.zip -d $DATA_DIR
fi

python3 electra-tamil/build_tamil_vocab.py

python3 electra-tamil/build_pretraining_dataset.py \
  --corpus-dir $DATA_DIR \
  --vocab-file $DATA_DIR/vocab.txt \
  --output-dir $DATA_DIR/pretrain_tfrecords \
  --max-seq-length 128 \
  --blanks-separate-docs False \
  --no-lower-case \
  --num-processes 5

python3 electra-tamil/run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name $MODEL_NAME \
