export DATA_DIR=data
export TRAIN_SIZE=100000000
export MODEL_NAME="electra-tamil" 
export CORPUS_URL=''
export ELECTRA_DIR=electra-tamil
virtualenv -p python3 ~/env/electra-tamil
sourc ~/env/electra-tamil/bin/activate

pip install tensorflow==1.15
pip install transformers==2.8.0
git clone https://github.com/vanangamudi/electra-tamil.git $ELECTRA_DIR


# make data directory: this is where model, pretrained records, vocab will live
mkdir -p $DATA_DIR

#download corpus
curl $CORPUS_URL -o $DATA_DIR/train_data.txt

python3 build_tamil_vocab.py

python3 $ELECTRA_DIR/build_pretraining_dataset.py \
  --corpus-dir $DATA_DIR \
  --vocab-file $DATA_DIR/vocab.txt \
  --output-dir $DATA_DIR/pretrain_tfrecords \
  --max-seq-length 128 \
  --blanks-separate-docs False \
  --no-lower-case \
  --num-processes 5

python3 $ELECTRA_DIR/run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name $MODEL_NAME \
