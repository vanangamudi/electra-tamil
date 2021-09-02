export ELECTRA_DIR=temp-test
export DATA_DIR=data
export MODEL_NAME=electra-tamil 

# make data directory: this is where model, pretrained records, vocab will live
mkdir -p $ELECTRA_DIR

cd $ELECTRA_DIR

mkdir -p $DATA_DIR

virtualenv -p python3 env
source env/bin/activate

pip3 install torch
pip3 install tensorflow-gpu==1.15.0
pip3 install transformers
pip3 install scipy sklearn
rm -rf electra-tamil
git clone https://github.com/vanangamudi/electra-tamil.git 

cd electra-tamil
git checkout -b chaii-finetune
cd ..

mkdir -p $DATA_DIR/finetuning_data/squad

echo $(pwd)

echo "finetuning..."
python3 electra-tamil/run_finetuning.py \
	--data-dir $DATA_DIR \
	--model-name $MODEL_NAME \
	--hparams '{"task_names": ["squad"] }'


python3 electra-tamil/run_finetuning.py \
	--data-dir $DATA_DIR \
	--model-name $MODEL_NAME \
	--hparams '{"do_train": false, "do_eval": true,  "task_names": ["squad"], "init_checkpoint": "$DATA_DIR/models/$MODEL_NAME/finetuning_models/squad_model_1"}'
