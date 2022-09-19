local train_data_path = "/quantus-nfs/users/v-weizhelin-DO-NOT-DELETE/user-movie-data-release/";
local validation_data_path = "";
local test_data_path = "";
local dated_file_list = [
    "user_movie_time_2021-05-13.tsv",
    "user_movie_time_2021-05-14.tsv",
    "user_movie_time_2021-05-15.tsv",
    "user_movie_time_2021-05-16.tsv",
    "user_movie_time_2021-05-17.tsv",
    "user_movie_time_2021-05-18.tsv",
    "user_movie_time_2021-05-19.tsv",
    "user_movie_time_2021-05-20.tsv",
    "user_movie_time_2021-05-21.tsv",
    "user_movie_time_2021-05-22.tsv",
    "user_movie_time_2021-05-23.tsv",
    "user_movie_time_2021-05-24.tsv",
    "user_movie_time_2021-05-25.tsv",
    "user_movie_time_2021-05-26.tsv",
    "user_movie_time_2021-05-27.tsv",
    "user_movie_time_2021-05-28.tsv",
    "user_movie_time_2021-05-29.tsv",
    "user_movie_time_2021-05-30.tsv",
    "user_movie_time_2021-05-31.tsv",
];
local dataset_name = "user-movie";
local dummy_train_data_path = "";
local dummy_validation_data_path = "";
local train_batch_size = 64;
local valid_batch_size = 64;
local test_batch_size = 64;
local valid_step_size = 1;
local save_interval = 1;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 10;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;

{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "platform_type": "pytorch",
  "model": "KGIN",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
  },
  "data_loader": {
    "type": "DataLoaderMoviesForNRMS",
    "dummy_dataloader": 0,
    "additional": {
        "train_data_path": train_data_path,
        "validation_data_path": validation_data_path,
        "test_data_path": test_data_path,
        "dataset_name": dataset_name,
        "file_list": dated_file_list,
        "dummy_train_data_path": dummy_train_data_path,
        "dummy_validation_data_path": dummy_validation_data_path,
    }
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "NRMSBERTExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch":-1,
    "save_interval":save_interval,
    "load_model_path": "",
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "load_model_path": "",
    "additional": {
        "multiprocessing": 4,
    },
  }
}