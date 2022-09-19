# Content-Aware-Knowledge-Enhanced-Meta-Preference-Networks-for-Recommendation
This is the official repository for CKMPN.

## Benchmarks
Work in progress.

## Data
We are applying for data release internally. We will release the data here once it is ready.

## Step-through Guidence

### Install requirements
```bash
pip install -r requirements.txt
```
Please note that the requirement file was dumped a long time ago. You may need to install some packages manually.

### Amazon-Book-Extended

#### KMPN
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/KMPN_books.jsonnet --mode train --experiment_name KMPN_AmazonBooks_MetaPreferences_64_PCA_0.5  --lr 0.001 --batch_size 65536 --epochs 2000 --scheduler linear --module_type PCA_DISTANCE_COR INDEPENDENT_RELATION_EMB --clipping 10 --node_dropout --negative_sampling_mode inversed_ratio --pca_ratio 0.5 --num_meta_preferences 64
```
#### NRMS-BERT
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/NRMS_BERT_books.jsonnet --mode train --experiment_name NRMS_BERT_Amazon_LR_0.0001_Layer_3_History_10_NoAttMask  --lr 0.0001 --batch_size 64 --test_batch_size 64 --epochs 1000 --clipping 10 --num_history 10 --freeze_transformer_layers 9
```
Pretrained model provided at `Experiments/NRMS_BERT_Amazon_LR_0.0001_Layer_3_History_10_NoAttMask`
#### CKMPN
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/KMPN_books.jsonnet --mode train --experiment_name KMPN_AmazonBooks_MetaPreferences_64_PCA_0.5_CrossSystemContrastiveLearning_0.1  --lr 0.001 --batch_size 65536 --epochs 2000 --scheduler linear --module_type PCA_DISTANCE_COR INDEPENDENT_RELATION_EMB NRMS_BERT_EMB NRMS_BERT_EMB_CROSS_SYSTEM --clipping 10 --node_dropout --negative_sampling_mode inversed_ratio --pca_ratio 0.5 --load_transformer_path ../Experiments/NRMS_BERT_Amazon_LR_0.0001_Layer_3_History_10_NoAttMask/train/saved_model/model_175.pth.tar  --dataset_name NRMS_BERT_Amazon_LR_0.0001_Layer_3_History_10_NoAttMask_EPOCH_175  --cross_system_loss_decay 0.1 --num_meta_preferences 64
```
Pretrained model provided at `Experiments/KMPN_AmazonBooks_MetaPreferences_64_PCA_0.5_CrossSystemContrastiveLearning_0.1`

### Microsoft-Movie-KG-Dataset  19days
#### KMPN
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/KMPN_movie_19d.jsonnet --mode train --enable_validation --experiment_name MOVIE_19d_ColdStart_KMPN_Batch_8192x8_lr_linear_5e-4_epoch_1000_PCA_0.5_MetaPreferences_64  --lr 0.0005 --batch_size 65536 --epochs 1000 --scheduler linear --dataset_name movie_dataset_19d_ColdStart --load_num_movies 50000 --load_num_users 200000 --split_cold_start --cold_start_ratio 0.03 --negative_sampling_mode inversed_ratio --clipping 10 --node_dropout --num_meta_preferences 64 --module_type INDEPENDENT_RELATION_EMB PCA_DISTANCE_COR --pca_ratio 0.5
```
#### NRMS-BERT
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/NRMS_BERT_movies_19d.jsonnet --mode train --enable_validation --experiment_name NRMS_BERT_Movie_19d_ColdStart_LR_0.0005_Layer_1_History_30_NoAttMask  --lr 0.0005 --batch_size 64 --test_batch_size 64 --epochs 1000 --dataset_name movie_dataset_19d_ColdStart --load_num_movies 50000 --load_num_users 200000 --num_history 30 --freeze_transformer_layers 11 --split_cold_start --cold_start_ratio 0.03
```
Pretrained model provided at `Experiments/NRMS_BERT_Movie_19d_ColdStart_LR_0.0005_Layer_1_History_30_NoAttMask`
#### CKMPN
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/KMPN_movie_19d.jsonnet --mode train --enable_validation --experiment_name MOVIE_19d_ColdStart_KMPN_Batch_8192x8_lr_linear_5e-4_epoch_1000_PCA_0.5_MetaPreferences_64_CrossProductSystem_0.01_valid_test  --lr 0.0005 --batch_size 65536 --epochs 1000 --scheduler linear --dataset_name movie_dataset_19d_ColdStart --load_num_movies 50000 --load_num_users 200000 --split_cold_start --cold_start_ratio 0.03 --negative_sampling_mode inversed_ratio --clipping 10 --node_dropout --num_meta_preferences 64 --module_type INDEPENDENT_RELATION_EMB PCA_DISTANCE_COR NRMS_BERT_EMB NRMS_BERT_EMB_CROSS_SYSTEM --pca_ratio 0.5 --cross_system_loss_decay 0.01 --load_transformer_path ../Experiments/NRMS_BERT_Movie_19d_ColdStart_LR_0.0005_Layer_1_History_30_NoAttMask/train/saved_model/model_46.pth.tar
```
#### Other Pretrained Models
NRMS-BERT pretrained on movie dataset (19d without ColdStart Split)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/NRMS_BERT_movies_19d.jsonnet --mode train --enable_validation --experiment_name NRMS_BERT_Movie_19d_LR_0.0005_Layer_1_History_30_NoAttMask  --lr 0.0005 --batch_size 64 --test_batch_size 64 --epochs 1000 --dataset_name movie_dataset_19d --load_num_movies 50000 --load_num_users 200000 --num_history 30 --freeze_transformer_layers 11
```
Pretrained model provided at `Experiments/NRMS_BERT_Movie_19d_LR_0.0005_Layer_1_History_30_NoAttMask`


### Microsoft-Movie-KG-Dataset  39days
#### Pretrained Models
NRMS-BERT pretrained on movie dataset (39d with ColdStart Split)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ../configs/NRMS_BERT_movies_39d.jsonnet --mode train --experiment_name NRMS_BERT_Movie_39d_ColdStart_LR_0.0005_Layer_1_History_30_NoAttMask  --lr 0.0005 --batch_size 64 --test_batch_size 64 --epochs 1000 --dataset_name movie_dataset_39d_ColdStart --load_num_movies 50000 --load_num_users 200000 --num_history 30 --freeze_transformer_layers 11 --split_cold_start --cold_start_ratio 0.03
```
Pretrained model provided at `Experiments/NRMS_BERT_Movie_39d_ColdStart_LR_0.0005_Layer_1_History_30_NoAttMask`


### Arguments
```
usage: main.py [-h] [--DATA_FOLDER DATA_FOLDER]
               [--EXPERIMENT_FOLDER EXPERIMENT_FOLDER] [--job_id JOB_ID]
               [--disable_cuda] [--device DEVICE [DEVICE ...]]
               [--module_type MODULE_TYPE [MODULE_TYPE ...]] [--mode MODE]
               [--reset] [--dummy_dataloader] [--regenerate_data]
               [--regenerate_transformer_data]
               [--experiment_name EXPERIMENT_NAME] [--fp16]
               [--load_best_model] [--load_epoch LOAD_EPOCH]
               [--load_model_path LOAD_MODEL_PATH]
               [--dataset_name [DATASET_NAME]]
               [--load_num_movies LOAD_NUM_MOVIES]
               [--load_num_users LOAD_NUM_USERS]
               [--load_transformer_path LOAD_TRANSFORMER_PATH]
               [--load_graph_path LOAD_GRAPH_PATH]
               [--test_num_evaluation TEST_NUM_EVALUATION]
               [--test_batch_size TEST_BATCH_SIZE]
               [--test_num_processes TEST_NUM_PROCESSES]
               [--test_evaluation_name TEST_EVALUATION_NAME] [--lr LR]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS]
               [--scheduler SCHEDULER] [--clipping CLIPPING]
               [--negative_sampling_mode NEGATIVE_SAMPLING_MODE] [--dim DIM]
               [--l2 L2] [--sim_regularity SIM_REGULARITY]
               [--inverse_r INVERSE_R] [--node_dropout]
               [--node_dropout_rate NODE_DROPOUT_RATE]
               [--mess_dropout MESS_DROPOUT]
               [--mess_dropout_rate MESS_DROPOUT_RATE]
               [--batch_test_flag BATCH_TEST_FLAG] [--channel CHANNEL]
               [--Ks KS [KS ...]] [--test_flag [TEST_FLAG]]
               [--n_factors N_FACTORS] [--ind IND]
               [--context_hops CONTEXT_HOPS] [--wl_test_iter WL_TEST_ITER]
               [--freeze_transformer_layers FREEZE_TRANSFORMER_LAYERS]
               [--num_negative_samples NUM_NEGATIVE_SAMPLES]
               [--num_history NUM_HISTORY] [--freeze_graph]
               [--split_model_on_gpus] [--pca_ratio PCA_RATIO]
               [--cross_system_loss_decay CROSS_SYSTEM_LOSS_DECAY]
               [--num_meta_preferences NUM_META_PREFERENCES] [--use_att_mask]
               [--extend_cbf_string EXTEND_CBF_STRING] [--split_cold_start]
               [--cold_start_ratio COLD_START_RATIO] [--enable_validation]
               config_json_file

positional arguments:
  config_json_file      The Configuration file in json format

optional arguments:
  -h, --help            show this help message and exit
  --DATA_FOLDER DATA_FOLDER
                        The path to data.
  --EXPERIMENT_FOLDER EXPERIMENT_FOLDER
                        The path to save experiments.
  --job_id JOB_ID
  --disable_cuda        Enable to run on CPU.
  --device DEVICE [DEVICE ...]
                        Specify GPU devices to use. -1 for default (all GPUs).
  --module_type MODULE_TYPE [MODULE_TYPE ...]
                        Select modules for models. See training scripts for
                        examples.
  --mode MODE           train/test
  --reset               Reset the corresponding folder under the
                        experiment_name
  --dummy_dataloader
  --regenerate_data     Regenerate movie dataset pre-processed data.
  --regenerate_transformer_data
                        Regenerate pre-trained user/item embeddings from NRMS-
                        BERT for CKMPN training.
  --experiment_name EXPERIMENT_NAME
                        Experiment will be saved under
                        /path/to/EXPERIMENT_FOLDER/$experiment_name$.
  --fp16                Not used.
  --load_best_model     Whether to load model_best.pth.tar.
  --load_epoch LOAD_EPOCH
                        Specify which epoch to load.
  --load_model_path LOAD_MODEL_PATH
                        Specify the path of model to load from
  --dataset_name [DATASET_NAME]
                        dataset name
  --load_num_movies LOAD_NUM_MOVIES
                        Limit of #movies to load (in pre-processing)
  --load_num_users LOAD_NUM_USERS
                        Limit of #users to load (in pre-processing)
  --load_transformer_path LOAD_TRANSFORMER_PATH
                        Load NRMS-BERT model for pre-extraction of user/item
                        embeddings.
  --load_graph_path LOAD_GRAPH_PATH
                        Not presented in this paper.
  --test_num_evaluation TEST_NUM_EVALUATION
  --test_batch_size TEST_BATCH_SIZE
  --test_num_processes TEST_NUM_PROCESSES
  --test_evaluation_name TEST_EVALUATION_NAME
  --lr LR               learning rate
  --batch_size BATCH_SIZE
                        batch size
  --epochs EPOCHS       number of epochs for training
  --scheduler SCHEDULER
                        which scheduler to use: [none, linear, cosine]
  --clipping CLIPPING   gradient clipping
  --negative_sampling_mode NEGATIVE_SAMPLING_MODE
                        choose from [inversed_ratio, ...]
  --dim DIM             embedding size
  --l2 L2               l2 regularization weight
  --sim_regularity SIM_REGULARITY
                        regularization weight for latent factor
  --inverse_r INVERSE_R
                        consider inverse relation or not
  --node_dropout        consider node dropout or not
  --node_dropout_rate NODE_DROPOUT_RATE
                        ratio of node dropout
  --mess_dropout MESS_DROPOUT
                        consider message dropout or not
  --mess_dropout_rate MESS_DROPOUT_RATE
                        ratio of node dropout
  --batch_test_flag BATCH_TEST_FLAG
                        use gpu or not
  --channel CHANNEL     hidden channels for model
  --Ks KS [KS ...]      Compute Metrics@K for K in Ks
  --test_flag [TEST_FLAG]
                        Specify the test type from {part, full}, indicating
                        whether the reference is done in mini-batch
  --n_factors N_FACTORS
                        number of latent factor for user favour
  --ind IND             Independence modeling: mi, distance, cosine
  --context_hops CONTEXT_HOPS
                        number of context hops
  --wl_test_iter WL_TEST_ITER
                        #iterations of WL test
  --freeze_transformer_layers FREEZE_TRANSFORMER_LAYERS
                        #layers to be fixed
  --num_negative_samples NUM_NEGATIVE_SAMPLES
                        #negative samples used in NRMS-BERT training.
  --num_history NUM_HISTORY
                        #user history used in training
  --freeze_graph
  --split_model_on_gpus
  --pca_ratio PCA_RATIO
                        Ratio of components to keep after PCA reduction before
                        calculating Distance Correlation Loss.
  --cross_system_loss_decay CROSS_SYSTEM_LOSS_DECAY
                        Loss weight of Cross System Contrastive Loss.
  --num_meta_preferences NUM_META_PREFERENCES
                        Number of meta preferences for modelling users.
  --use_att_mask        NRMS-BERT: False (recommended): randomly sample
                        #history from user histories regardless of
                        duplications; True: for users with less than #history,
                        put attention masks in attention pooling stage.
  --extend_cbf_string EXTEND_CBF_STRING
  --split_cold_start    Whether to split a Cold Start user set from the
                        original dataset.
  --cold_start_ratio COLD_START_RATIO
                        Ratio of users to be put into Cold Start user set.
  --enable_validation   Whether to enable validation (using validation set).
                        Only available in movie datasets.

```


## BPRMF/CKE/KGAT (baseline)
For Amazon-Book dataset, the official repository of KGAT can be used for result reproduction.
We noticed a bug in `Model/utility/metrics.py`, and we have fixed it.
Please copy `src/baselines/KGAT-movies/Model/utility/metrics.py` to the cloned official GAT repository for reproduction.

For the movie dataset, we provided full codes for reproduction, under `src/baselines/KGAT-movies`.

### Baselines on Amazon dataset
BPRMF
```
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 0 | tee bprmf_amazon_baseline_log.log
# Save for KGAT
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag -1 --pretrain 1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 0 --report 0
```
CKE
```
python Main.py --model_type cke --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 1 | tee cke_amazon_baseline_log.log
```
KGAT
```
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 3 | tee kgat_amazon_baseline_official_weights_log.log
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 50 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 3 | tee kgat_amazon_baseline_official_weights_log.log
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 3 | tee kgat_amazon_baseline_my_weights_log.log
```
Run test
```
# read results
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 0
python Main.py --model_type cke --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 1
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --gpu_id 3
```
### Baselines on movie 19d cold start dataset
BPRMF
```
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_19_days_cold_start --gpu_id 0 | tee bprmf_movie_19d_cold_start_log.log
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag -1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_19_days_cold_start --gpu_id 0
```
CKE
```
python Main.py --model_type cke --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_19_days_cold_start --gpu_id 1 | tee cke_movie_19d_cold_start_log.log
```
KGAT
```
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_19_days_cold_start --gpu_id 3 | tee kgat_movie_19d_cold_start_log.log
```
### Baselines on movie 39d cold start dataset
BPRMF
```
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 0 | tee bprmf_movie_39d_cold_start_log.log
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag -1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 0 | tee bprmf_movie_39d_cold_start_log.log
```
CKE
```
python Main.py --model_type cke --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 1 | tee cke_movie_39d_cold_start_log.log
```
KGAT
```
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 3 | tee kgat_movie_39d_cold_start_log.log
```

Run test
```
python Main.py --model_type bprmf --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 0
python Main.py --model_type cke --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 1
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 1000 --verbose 1 --save_flag 1 --pretrain 1 --report 0 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True --movie_dataset_name 50kfrequent_movies_200kfrequent_users_compact_39_days_cold_start --gpu_id 3
```

## Computing Facilities
KGIN/KMPN/NRMS-BERT/CKMPN were run on:
```
cluster with 8 NVIDIA A100 (40 GB GPU Memory)
Cuda compilation tools, release 11.1
```
KGAT/BPRMF/CKE were run on:
```
cluster with 8 NVIDIA V100 (24 GB GPU Memory)
Cuda compilation tools, release 10.0
```

## Remarks

1. The code is derived from our research codebase by removing dependencies of the internal corp data and service. Therefore, some settings are not exactly matched to the code for producing the numbers reported in the paper.
2. There is a pytroch implementation for KGAT [here](https://github.com/LunaBlack/KGAT-pytorch) which we tried to migrate to our codebase in the preliminary stage of our research. However, it seems that the implementation (either ours or the original one) is not correct. The corresponding trainer is in `src\train_executor\backup\KGAT_TF_executor.py.backup` for your interest.
3. The codebase is still under development. But given that the first author has left Microsoft and not able to connect to the original research repository and service, we recommend you to use the benchmark results as a reference instead of directly using the codebase in your research.
