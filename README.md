## Requirements

* This work was tested with PyTorch 1.3.1, CUDA 10.1, Python 3.6
* Build the required dependencies from the following command

```shell
pip install -r requirements.txt
```

## Usage

* Download pre-trained word vectors

```shell
wget -P data/ http://nlp.stanford.edu/data/glove.6B.zip && unzip -d data/ data/glove.6B.zip 
```

* Preprocess data

```shell
# for MultiWoz
python utils/utils_newAgentUser.py --data_name newSpUser
# for DailyDialogue
python utils/utils_newDailyDialog.py --data_name DailyUser
# for CCPE
python utils/utils_newCCPE.py --data_name CCPEAgent
# preprocess for generator
sh run_preprocess_data.sh
```

* Training for Generator

```shell
# LSTM + Attention model for MultiWoz and DailyDialog
sh train_LSTM_attn.sh newAgentUser newSpUser
sh train_LSTM_attn.sh newDailyDialog DailyUser

# LSTM + Attention + GloVe model for MultiWoz and DailyDialog
sh train_LSTM_attn_GloVe.sh newAgentUser newSpUser
sh train_LSTM_attn_GloVe.sh newDailyDialog DailyUser

# LSTM model for MultiWoz and DailyDialog
sh train_LSTM.sh newAgentUser newSpUser
sh train_LSTM.sh newDailyDialog DailyUser

# all LSTM model for CCPE
sh train_LSTM_all_for_CCPE.sh newCCPE CCPEAgent
```

* Inference for Generator

```shell
# sh xxx.sh [dataset_name] [data_name] [model_name]
# checkpoint path: nmt/[dataset_name]/[data_name]/[model_name]

# Example
# LSTM model for MultiWoz, DailyDialog and CCPE
sh infer_LSTM.sh newAgentUser newSpUser agent-woattn-model_step_90000.pt user-woattn-model_step_90000.pt
sh infer_LSTM.sh newDailyDialog DailyUser agent-woattn-model_step_100000.pt user-woattn-model_step_90000.pt
sh infer_LSTM_CCPE.sh newCCPE CCPEAgent agent-woattn-model_step_2600.pt user-woattn-model_step_2600.pt

# LSTM + Attention model for MultiWoz, DailyDialog and CCPE
sh infer_LSTM_attn.sh newAgentUser newSpUser agent-glove-model_step_80000.pt user-glove-model_step_80000.pt
sh infer_LSTM_attn.sh newDailyDialog DailyUser agent-glove-model_step_80000.pt user-glove-model_step_90000.pt
sh infer_LSTM_attn_CCPE.sh newCCPE CCPEAgent agent-glove-model_step_3600.pt user-glove-model_step_3400.pt

# LSTM + Attention + GloVe model for MultiWoz, DailyDialog and CCPE
sh infer_LSTM_attn_GloVe.sh newAgentUser newSpUser agent-origin-model_step_70000.pt user-origin-model_step_80000.pt
sh infer_LSTM_attn_GloVe.sh newDailyDialog DailyUser agent-origin-model_step_100000.pt user-origin-model_step_80000.pt
sh infer_LSTM_attn_GloVe_CCPE.sh newCCPE CCPEAgent agent-origin-model_step_2000.pt user-origin-model_step_2600.pt
```

* Evaluate for Generator

```shell
# for MultiWoz 
sh eval_BLEU.sh newAgentUser newSpUser
# for DailyDialogue
sh eval_BLEU.sh newDailyDialog DailyUser
# for CCPE
sh eval_BLEU.sh newCCPE CCPEAgent
```

* Prepare data for Classifier

```shell
sh run_prepare_data.sh
```

* Training, Validation, Testing for Text-CNN

```shell
# run Baseline
# for MultiWoz
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_eval --do_train --single_mode
# for DailyDialogue
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name DailyUser --do_eval --do_train --single_mode
# for CCPE
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name CCPEAgent --do_eval --do_train --single_mode
```

```shell
# run ModCNN (Note: "Mod" means modified)
#CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#--addition_name ModCNN \
#-bsz [batch size] -kn [kernel num] \
#-dr [dr] -ksz [kernel sizes]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_eval --do_train --addition_name ModCNN -bsz=64 -kn=100 -dr=0.5 -ksz 4 5 6
```

```shell
# run Text-CNN from checkpoint 
#CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py \
#--data_name [data_name] \
#--do_eval --ckpt [checkpoint_path]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_eval --ckpt ckpt.pt
```

* Training, Validation, Testing for Bi-GRU

```shell
# run Baseline
# for MultiWoz
CUDA_VISIBLE_DEVICES=0 python trainRNN.py --data_name newSpUser --do_eval --do_train --single_mode
# for DailyDialogue
CUDA_VISIBLE_DEVICES=0 python trainRNN.py --data_name DailyUser --do_eval --do_train --single_mode
# for CCPE
CUDA_VISIBLE_DEVICES=0 python trainRNN.py --data_name CCPEAgent --do_eval --do_train --single_mode
```

```shell
# run ModRNN
#CUDA_VISIBLE_DEVICES=0 python trainRNN.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#--addition_name ModRNN \
#-bsz [batch size] -hdim [hidden size] \
#-dr [dr]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_eval --do_train --addition_name ModRNN -bsz 64 -dr 0.6 -hdim 600
```

```shell
# run ModRNN from checkpoint 
#CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py \
#--data_name [data_name] \
#--do_eval --ckpt [checkpoint_path]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainRNN.py --data_name newSpUser --do_eval --ckpt ckpt.pt
```

* Training, Validation, Testing for BERT

```shell
# run Baseline
# for MultiWoz
sh train_BERT.sh newSpUser baseline
# for DailyDialogue
sh train_BERT.sh DailyUser baseline
# for CCPE
sh train_BERT.sh CCPEAgent baseline
```

```shell
# run ModBERT
# for MultiWoz
sh train_BERT.sh newSpUser ModBERT
# for DailyDialogue
sh train_BERT.sh DailyUser ModBERT
# for CCPE
sh train_BERT.sh CCPEAgent ModBERT
```

* Run Random Classifier on Test set

```shell
python random_classifier.py --data_name newSpUser
python random_classifier.py --data_name DailyUser
python random_classifier.py --data_name CCPEAgent
```

* Test the performance of different generation effects on Classifier

```shell
python trainTextCNN.py --do_eval --data_name [data name] --ckpt [best_ckpt.pt] --nmt_name woattn # woattn means LSTM without attention
python trainTextCNN.py --do_eval --data_name [data name] --ckpt [best_ckpt.pt] --nmt_name origin # origin means LSTM with attention
python trainTextCNN.py --do_eval --data_name [data name] --ckpt [best_ckpt.pt] --nmt_name glove # glove means LSTM+attention+GloVe
```

* Sample results from generated text

```shell
# for MultiWoz
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_sample --ckpt ckpt_best.pt
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name newSpUser --do_sample --ckpt ckpt_basline.pt --single_mode
python sample_generator.py --data_name newSpUser
# the output is in output folder...
# for DailyDialogue and CCPE
# same as MultiWoz
```




