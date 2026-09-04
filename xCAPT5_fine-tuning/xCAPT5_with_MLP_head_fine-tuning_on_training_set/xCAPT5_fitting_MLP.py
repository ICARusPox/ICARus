"""
This Python script running xCAPT5 has to be run in the corresponding
virtual environment.

Also note that this implementation accepts input from the command line.
"""
import os
import argparse

import torch
from torch import nn
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb

from xCAPT5_utils import (get_T5_model, read_fasta, get_embeddings,
    save_embeddings, pad, leaky_relu, multi_cnn)

try:
   os.mkdir("protT5")
   os.mkdir("protT5/protT5_checkpoint")
   os.mkdir("protT5/sec_struct_checkpoint")
   os.mkdir("protT5/output")
except FileExistsError:
   print("\nDirectories have already been created.\n")

parser = argparse.ArgumentParser()

parser.add_argument("train_pairs_file", type=str)
parser.add_argument("val_pairs_file", type=str)
parser.add_argument("fasta_file", type=str)
parser.add_argument("MCAPST5_model", type=str)
parser.add_argument("seed", type=int)
parser.add_argument("output")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--hid_dim", type=int, default=16)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--ckpt_interval", type=int, default=5)

args = parser.parse_args()

if "pan" in args.MCAPST5_model:
   MCAPST5_ckpt_name = "Pan"
elif "sled" in args.MCAPST5_model:
   MCAPST5_ckpt_name = "Sled"
else:
   raise ValueError(
      "MCAPST5 checkpoint name must contain the name of one of the two "
      "data sets (Pan or Sled)."
   )

run = wandb.init(
   entity="mlid",
   project="MLP_head_for_PPI_prediction_ordinary_loss",
   name=f"ordinary_loss_{MCAPST5_ckpt_name}_{args.epochs}_epochs_seed_{args.seed}",
   # Track hyperparameters and run metadata
   config={
      "learning_rate": args.lr,
      "hidden_dim": args.hid_dim,
      "weight_decay": args.weight_decay,
      "dropout": args.dropout,
      "epochs": args.epochs,
      "architecture": "MLP"
   }
)

# Conveniently enough, xCAPT5 requires the same input as SENSE-PPI, i.e.
# a FASTA file encompassing all proteins as well as a TSV file listing
# all interaction pairs to investigate
# The required input format is also identical with that of SENSE-PPI,
# i.e. the FASTA file headers exclusively contain the protein ID and the
# TSV files comprises combinations of those protein IDs
train_pair_path = args.train_pairs_file
val_pair_path = args.val_pairs_file
seq_path = args.fasta_file

# The implementation provided by the authors expects the TSV file to
# have a third column harbouring interaction labels (i.e. 1 for True and
# 0 for False), irrespective of whether they are known or not
# Therefore, in case the TSV used does not already possess a third
# column, one is introduced having zero as its values
train_pair_df = pd.read_csv(train_pair_path, sep="\t", header=None)
if train_pair_df.shape[1] < 3:
   train_pair_df["Label"] = 0
   train_pair_df.to_csv(
      train_pair_path,
      sep="\t",
      header=False,
      index=False
   )

val_pair_df = pd.read_csv(val_pair_path, sep="\t", header=None)
if val_pair_df.shape[1] < 3:
   val_pair_df["Label"] = 0
   val_pair_df.to_csv(
      val_pair_path,
      sep="\t",
      header=False,
      index=False
   )

# Define the embedding type
# Possible options include embeddings per residue (yields a Lx1024
# matrix per protein with L being the protein's length) as well as
# embeddings protein (yields a 1024-dimensional vector per protein,
# irrespective of its length)
per_residue = True 
per_residue_path = "./protT5/output/per_residue_embeddings.h5"

per_protein = False
per_protein_path = "./protT5/output/per_protein_embeddings.h5"

# For some strange reason to still be fathomed out, secondary structures
# are somehow involved in xCAPT5
sec_struct = False
sec_struct_path = "./protT5/output/ss3_preds.fasta"

if per_residue:
   embedding_path = per_residue_path
elif per_protein:
   embedding_path = per_protein_path

assert (
    per_protein is True or per_residue is True or sec_struct is True
), print(
     "Minimally, you need to active per_residue, per_protein or "
     "sec_struct. (or any combination)"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")


if os.path.isfile(per_residue_path) == False:
    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = get_T5_model(device=device)

    # Load example fasta.
    seqs = read_fasta( seq_path )

    for id, seq in seqs.items():
        if len(seq) > 1200:
            seqs[id] = seq[:1200]


    # Compute embeddings and/or secondary structure predictions
    results = get_embeddings(
        model, tokenizer, seqs, per_residue, per_protein, sec_struct,
        device
    )

    # Store per-residue embeddings
    if per_residue:
      save_embeddings(results["residue_embs"], per_residue_path)
    if per_protein:
      save_embeddings(results["protein_embs"], per_protein_path)
else:
    print("Already have the embedding file")


### Setting RAM GPU for training growth 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. When disabled,
# no binary code is added to or retrieved from the cache.
os.environ['CUDA_CACHE_DISABLE'] = '0' # orig is 0

# When set to 1, forces the device driver to ignore any binary code embedded in an application 
# (see Application Compatibility) and to just-in-time compile embedded PTX code instead.
# If a kernel does not have embedded PTX code, it will fail to load. This environment variable can be used to
# validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application 
# forward compatibility with future architectures.
os.environ['CUDA_FORCE_PTX_JIT'] = '1'# no orig


os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'

os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"



# =================================================
mixed_precision.set_global_policy('mixed_float16')


## Set constant hyperparameters
BATCH_SIZE = 64
seq_size = 1200
dim = 1024


# Read interaction data set to DataFrame
print("Load the train pair dataset file")
train_pair_dataframe = pd.read_csv(train_pair_path, sep='\t', header=None)
train_pair_array  = train_pair_dataframe.to_numpy()
train_pair_dataframe = pd.DataFrame(train_pair_array, columns=['p1', 'p2', 'label'])
train_pair_dataframe['label'] = train_pair_dataframe['label'].astype('float16') 
train_pair_dataframe['p1'] = train_pair_dataframe['p1'].str.replace(".","_")
train_pair_dataframe['p2'] = train_pair_dataframe['p2'].str.replace(".","_")

# Do the same for the validation set
print("Load the validation pair data set file")
val_pair_dataframe = pd.read_csv(val_pair_path, sep='\t', header=None)
val_pair_array = val_pair_dataframe.to_numpy()
val_pair_dataframe = pd.DataFrame(val_pair_array, columns=['p1', 'p2', 'label'])
val_pair_dataframe['label'] = val_pair_dataframe['label'].astype('float16')
val_pair_dataframe['p1'] = val_pair_dataframe['p1'].str.replace(".","_")
val_pair_dataframe['p2'] = val_pair_dataframe['p2'].str.replace(".","_")


# Read the embedding matrix
embedding_matrix= h5py.File(embedding_path, 'r')
protein_keys = list(embedding_matrix.keys())
embedding_dict = dict()

for key in protein_keys:
  embedding_dict[key] = np.array(embedding_matrix[key])


def train_func(i):
    i = i.numpy() # Decoding from the EagerTensor object
    x1= pad(embedding_dict[train_pair_dataframe['p1'][i]])
    x2= pad(embedding_dict[train_pair_dataframe['p2'][i]])
    y = train_pair_dataframe['label'][i]
    return x1, x2, y

def val_func(i):
   i = i.numpy() # Decoding from the EagerTensor object
   x1 = pad(embedding_dict[val_pair_dataframe['p1'][i]])
   x2 = pad(embedding_dict[val_pair_dataframe['p2'][i]])
   y = val_pair_dataframe['label'][i]
   return x1, x2, y

def _fixup_shape(x1, x2, y):
    x1.set_shape((seq_size, dim))
    x2.set_shape((seq_size, dim)) 
    y.set_shape(()) 

    return (x1, x2), y


# Create the training as well as the evaluation data set object
train_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(train_pair_dataframe)), tf.int64
).map(
   lambda i: tf.py_function(
       func=train_func, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
   ),
   num_parallel_calls=tf.data.AUTOTUNE
).map(_fixup_shape).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(val_pair_dataframe)), tf.int64
).map(
   lambda i: tf.py_function(
      func=val_func, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
   ),
   num_parallel_calls=tf.data.AUTOTUNE
).map(_fixup_shape).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Architecture of MCAPS
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def lisht(x):
    return x * tf.math.tanh(x)

def rrelu(x):
    return tf.nn.relu(x)  # approximation (training-time randomness usually not critical here)

get_custom_objects().update({'leaky_relu': leaky_relu})
get_custom_objects().update({'mish': mish})
get_custom_objects().update({'lisht': lisht})
get_custom_objects().update({'rrelu': rrelu})

model = multi_cnn()
model.summary()

# Load checkpoint
mcapst5_model = args.MCAPST5_model

@tf.keras.utils.register_keras_serializable()
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._loaded = None
        self._infer = None

    def build(self, input_shape):
        # Load SavedModel lazily (important for serialization)
        self._loaded = tf.saved_model.load(self.model_path)
        self._infer = self._loaded.signatures["serving_default"]

    def call(self, inputs):
        seq1, seq2 = inputs
        outputs = self._infer(seq1=seq1, seq2=seq2)
        return list(outputs.values())[0]  # extract tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_path": self.model_path
        })
        return config

model = tf.keras.models.load_model(mcapst5_model, compile=False)
model.summary()

shape_x = model.output.shape[1]

# Transform pairs matrix
X_train = model.predict(train_dataset)
y_train = train_pair_dataframe['label'].values.reshape(-1,)

# Do the same for the validation set
X_val = model.predict(validation_dataset)
y_val = val_pair_dataframe['label'].values.reshape(-1,)

# Convert the data set to make it compatible with PyTorch
X_train = X_train.astype(np.float32)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = X_val.astype(np.float32)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)


# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Define the MLP head
mlp_model = nn.Sequential(
   nn.Linear(shape_x, args.hid_dim),
   nn.ReLU(),
   nn.Dropout(args.dropout),
   nn.Linear(args.hid_dim, 1)
)

mlp_model = mlp_model.to(device)

# Define the optimizer and the loss
optimizer = torch.optim.AdamW(
   mlp_model.parameters(),
   lr=args.lr,
   weight_decay=args.weight_decay
)
loss = torch.nn.BCEWithLogitsLoss()

# Create DataLoaders
train_dataset_torch = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset_torch = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(
   train_dataset_torch,
   batch_size=100,
   shuffle=True
)
val_loader = torch.utils.data.DataLoader(
   val_dataset_torch,
   batch_size=100,
   shuffle=False
)


# Create the directory to store the MLP checkpoints in
ckpt_dir = f"MLP_ckpts_{MCAPST5_ckpt_name}_seed_{args.seed}"

if not os.path.exists(ckpt_dir):
   os.makedirs(ckpt_dir)
if os.path.exists(ckpt_dir):
   for file_name in os.listdir(ckpt_dir):
      os.remove(os.path.join(ckpt_dir, file_name))


best_val_loss = float("inf")

for epoch in range(args.epochs):
   mlp_model.train()
   train_loss = 0.0

   for xb, yb in train_loader:
      optimizer.zero_grad()

      logits = mlp_model(xb).view(-1)
      loss_val = loss(logits, yb)

      loss_val.backward()
      optimizer.step()

      train_loss += loss_val.item()
   
   train_loss /= len(train_loader)

   # ===== Validation =====
   mlp_model.eval()
   val_loss = 0.0
   all_preds = []
   all_labels = []

   with torch.no_grad():
      for xb, yb in val_loader:
         logits = mlp_model(xb).view(-1)
         loss_val = loss(logits, yb)

         val_loss += loss_val.item()

         probs = torch.sigmoid(logits)
         all_preds.append(probs.cpu().numpy())
         all_labels.append(yb.cpu().numpy())

   val_loss /= len(val_loader)

   all_preds = np.concatenate(all_preds)
   all_labels = np.concatenate(all_labels)

   mean_prob = all_preds.mean()
   max_prob = all_preds.max()
   min_prob = all_preds.min()

   # ===== Metrics =====
   try:
      roc_auc = roc_auc_score(all_labels, all_preds)
      pr_auc = average_precision_score(all_labels, all_preds)
   except:
      roc_auc, pr_auc = np.nan, np.nan
   
   # ===== Logging =====
   wandb.log({
      "epoch": epoch,
      "train_loss": train_loss,
      "val_loss": val_loss,
      "roc_auc": roc_auc,
      "pr_auc": pr_auc,
      "mean_prob": mean_prob,
      "max_prob": max_prob,
      "min_prob": min_prob
   })

   print(
      f"Epoch {epoch}: train_loss={train_loss:.4f}, "
      f"val_loss={val_loss:.4f}, ROC-AUC={roc_auc:.4f}"
   )

   # ===== Checkpointing =====
   if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(
         mlp_model.state_dict(),
         os.path.join(ckpt_dir, "best_model.pt")
      )
   
   if epoch % args.ckpt_interval == 0:
      torch.save(
         mlp_model.state_dict(),
         os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
      )

# Don't forget to save the fitted model!
torch.save(
   mlp_model.state_dict(),
   f"MLP_{args.output}_ordinary_loss_seed_{args.seed}.pt"
)

# Finish the run and upload any remaining data
run.finish()