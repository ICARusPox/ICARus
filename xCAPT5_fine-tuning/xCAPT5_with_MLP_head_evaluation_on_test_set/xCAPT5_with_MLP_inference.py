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

from xCAPT5_utils import (get_T5_model, read_fasta,get_embeddings,
    save_embeddings, pad, leaky_relu, multi_cnn)

try:
   os.mkdir("protT5")
   os.mkdir("protT5/protT5_checkpoint")
   os.mkdir("protT5/sec_struct_checkpoint")
   os.mkdir("protT5/output")
except FileExistsError:
   print("\nDirectories have already been created.\n")

parser = argparse.ArgumentParser()

parser.add_argument("pairs_file", type=str)
parser.add_argument("fasta_file", type=str)
parser.add_argument("MCAPST5_model", type=str)
parser.add_argument("MLP_model_ckpt", type=str)
parser.add_argument("seed", type=int)
parser.add_argument("output")
parser.add_argument("--hid_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--ppi_prior", type=float, default=None)

args = parser.parse_args()

# Conveniently enough, xCAPT5 requires the same input as SENSE-PPI, i.e.
# a FASTA file encompassing all proteins as well as a TSV file listing
# all interaction pairs to investigate
# The required input format is also identical with that of SENSE-PPI,
# i.e. the FASTA file headers exclusively contain the protein ID and the
# TSV files comprises combinations of those protein IDs
pair_path = args.pairs_file
seq_path = args.fasta_file

# The implementation provided by the authors expects the TSV file to
# have a third column harbouring interaction labels (i.e. 1 for True and
# 0 for False), irrespective of whether they are known or not
# Therefore, in case the TSV used does not already possess a third
# column, one is introduced having zero as its values
pair_df = pd.read_csv(pair_path, sep="\t", header=None)
if pair_df.shape[1] < 3:
   pair_df["Label"] = 0
   pair_df.to_csv(
      pair_path,
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
print("Load the PPI pairs file")
pair_dataframe = pd.read_csv(pair_path, sep="\t", header=None)
pair_array = pair_dataframe.to_numpy()
pair_dataframe = pd.DataFrame(pair_array, columns=['p1', 'p2', 'label'])
pair_dataframe['label'] = pair_dataframe['label'].astype('float16')
pair_dataframe['p1'] = pair_dataframe['p1'].str.replace(".","_")
pair_dataframe['p2'] = pair_dataframe['p2'].str.replace(".","_")


# Read the embedding matrix
embedding_matrix= h5py.File(embedding_path, 'r')
protein_keys = list(embedding_matrix.keys())
embedding_dict = dict()

for key in protein_keys:
  embedding_dict[key] = np.array(embedding_matrix[key])


def func(i):
    i = i.numpy() # Decoding from the EagerTensor object
    x1= pad(embedding_dict[pair_dataframe['p1'][i]])
    x2= pad(embedding_dict[pair_dataframe['p2'][i]])
    y = pair_dataframe['label'][i]
    return x1, x2, y

def _fixup_shape(x1, x2, y):
    x1.set_shape((seq_size, dim))
    x2.set_shape((seq_size, dim)) 
    y.set_shape(()) 

    return (x1, x2), y


# Create the test data set object
test_dataset = tf.data.Dataset.from_generator(
   lambda: range(len(pair_dataframe)), tf.uint64
).map(
   lambda i: tf.py_function(
       func=func, inp=[i], Tout=[tf.float16, tf.float16, tf.float16]
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

# Load MCAPST5 checkpoint
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

# Transform pairs matrix
pred = model.predict(test_dataset)

shape_x = model.output.shape[1]

pred = pred.astype(np.float32)


# Define the architecture of the MLP head, load the checkpoint and move
# the model to device
mlp_model = nn.Sequential(
   nn.Linear(shape_x, args.hid_dim),
   nn.ReLU(),
   nn.Dropout(args.dropout),
   nn.Linear(args.hid_dim, 1)
)

mlp_model.load_state_dict(
   torch.load(args.MLP_model_ckpt, map_location=device)
)

mlp_model = mlp_model.to(device)
mlp_model.eval()

all_logits = []
all_probs = []

if args.ppi_prior is not None:
    prior_logit = torch.log(
        torch.tensor(args.ppi_prior / (1 - args.ppi_prior), device=device)
    )
else:
    prior_logit = None

# Finally, run inference
for i in range(0, pred.shape[0], BATCH_SIZE):
    batch_np = pred[i:i + BATCH_SIZE]

    X = torch.from_numpy(batch_np).to(device)

    with torch.no_grad():
        logits = mlp_model(X).view(-1)

        if args.ppi_prior is not None:
            logits += prior_logit

        probs = torch.sigmoid(logits)

    # Move back to CPU to free GPU memory
    all_logits.append(logits.cpu())
    all_probs.append(probs.cpu())

logits = torch.cat(all_logits)
probs = torch.cat(all_probs)


# Save the prediction results to a TSV file
# In the TSV file, the first and second column contains the first and
# second interaction partner, respectively, whereas the third column
# stores the predicted probability
# Additionally, a fourth column lists the labels corresponding to the
# predicted probabilities
# Generate the labels using a threshold of 0.5
predicted_labels = (probs >= 0.5).int()

probs_np = probs.numpy()
logits_np = logits.numpy()
labels_np = predicted_labels.numpy()

results_df = pd.DataFrame(data={
   "protein_1": pair_dataframe["p1"],
   "protein_2": pair_dataframe["p2"],
   "logits": logits_np,
   "interaction_probability": probs_np,
   "label": labels_np
})

results_df.to_csv(
   f"xCAPT5_with_MLP_interaction_probs_{args.output}_"
   f"seed_{args.seed}_PPI_prior_{args.ppi_prior}.tsv",
   sep="\t",
   index=False
)