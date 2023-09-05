from easydict import EasyDict as edict

config = edict()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------

# Recognition val targets
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw', 'cplfw']

# Recognition data
config.rec = "<your path>/data/faces_emore/"  #### Path for the training dataset
config.num_classes = 85742
config.num_image = 5822653

# Data loading settings
config.batch_size = 128
config.num_workers = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------

config.network = "swin_t"

config.embedding_size = 512

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------

# Resume
config.resume = False

# Epoch num
config.warmup_epoch = 5
config.num_epoch = 40

# For SGD
#config.optimizer = "sgd"
#config.lr = 0.1
#config.momentum = 0.9
#config.weight_decay = 5e-4

# For AdamW
config.optimizer = "adamw"
config.lr = 5e-4
config.weight_decay = 0.05

# Learning rate
config.lr_name = 'cosine'
config.warmup_lr = 5e-7
config.min_lr = 5e-6
config.decay_epoch = 10 # Epoch interval to decay LR, used in StepLRScheduler
config.decay_rate = 0.1 # LR decay rate, used in StepLRScheduler

# loss
config.margin_list = (1.0, 0.0, 0.4)
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0

# Others
config.fp16 = True
config.dali = False # For Large Sacle Dataset, such as WebFace42M
config.seed = 2048

# -----------------------------------------------------------------------------
# Output and Saving
# -----------------------------------------------------------------------------

config.save_all_states = True
config.output = "<your path>/output" ####Path for Output

config.verbose = 2000
config.frequent = 10








