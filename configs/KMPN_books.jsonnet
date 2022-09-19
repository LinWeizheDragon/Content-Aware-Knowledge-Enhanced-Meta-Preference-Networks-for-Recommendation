local train_data_path = "../data/amazon-book";
local validation_data_path = "../data/amazon-book";
local test_data_path = "../data/amazon-book";
local dataset_name = "amazon-book";
local dummy_train_data_path = "";
local dummy_validation_data_path = "";
local train_batch_size = 2048;
local valid_batch_size = 8196;
local test_batch_size = 8196;
local valid_step_size = 10;
local save_interval = 200;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local graph_lr = 1e-4;
local gradient_accumulation_steps = 1;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;

{
  "DATA_FOLDER": "",
  "EXPERIMENT_FOLDER": "",
  "TENSORBOARD_FOLDER": "",
  "platform_type": "pytorch",
  "model": "KMPN",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
  },
  "data_loader": {
    "type": "DataLoaderAmazonBooks",
    "dummy_dataloader": 0,
    "additional": {
        "train_data_path": train_data_path,
        "validation_data_path": validation_data_path,
        "test_data_path": test_data_path,
        "dataset_name": dataset_name,
        "dummy_train_data_path": dummy_train_data_path,
        "dummy_validation_data_path": dummy_validation_data_path,
    }
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "KMPNExecutor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "graph_lr":graph_lr,
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