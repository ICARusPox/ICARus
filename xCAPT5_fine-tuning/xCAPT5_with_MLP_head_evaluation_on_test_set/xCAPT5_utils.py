import time

import h5py
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, Conv1D, GlobalMaxPooling1D, \
    AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs


def get_T5_model(device):
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, device,
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    if sec_struct:
      sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict(),
               "sec_structs" : dict() 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer(seqs, add_special_tokens=True, padding=True, return_tensors="pt")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            if sec_struct: # in case you want to predict secondary structure from embeddings
              d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                if sec_struct: # get classification results
                    results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results


def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None


def pad(rst, length=1200, dim=1024):
    if len(rst) > length:
        return rst[:length]
    elif len(rst) < length:
        return np.concatenate((rst, np.zeros((length - len(rst), dim))))
    return rst


def leaky_relu(x, alpha = .2):
   return tf.keras.backend.maximum(alpha*x, x)


def multi_cnn(seq_size=1200, dim=1024, learning_rate=1e-3):
    DEPTH = 5
    WIDTH = 3
    POOLING_SIZE = 4
    FILTERS = 50
    KERNEL_SIZE = 2
    DEPTH_DENSE1 = 3
    DEPTH_DENSE2 = 2
    DROPOUT = DROPOUT1 = DROPOUT2 = 0.05
    DROPOUT_SPATIAL= 0.15
    ACTIVATION = 'swish'
    ACTIVATION_CNN = 'swish'
    INITIALIZER = 'glorot_normal'
    
    def BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2):
        c1 = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION_CNN, padding='same')
        x1 = c1(LAYER_IN1)
        x2 = c1(LAYER_IN2)

        g1 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x1),GlobalAveragePooling1D()(x1)]))
        a1 = GlobalAveragePooling1D()(x1)
        g2 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x2),GlobalAveragePooling1D()(x2)]))
        a2 = GlobalAveragePooling1D()(x1)

        x1 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x1), AveragePooling1D(POOLING_SIZE)(x1)]))
        x2 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x2), AveragePooling1D(POOLING_SIZE)(x2)]))

        return x1, x2, g1, g2, a1, a2

    def BlockCNN_multi(POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2, WIDTH):
      X1 = []
      X2 = []
      G1 = []
      G2 = []
      A1 = []
      A2 = []
      for i in range(2, 2+WIDTH):
        x1, x2, g1, g2, a1, a2 = BlockCNN_single(i, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2)
        X1.append(x1)
        X2.append(x2)
        G1.append(g1)
        G2.append(g2)
        A1.append(a1)
        A2.append(a2)
      x1 = concatenate(X1)
      x2 = concatenate(X2)
      g1 = GlobalMaxPooling1D()(x1)
      g2 = GlobalMaxPooling1D()(x2)
      return x1, x2, g1, g2

    def BlockCNN_single_deep(KERNEL_SIZE, POOLING_SIZE, DEPTH, FILTERS, LAYER_IN1, LAYER_IN2):
      X1 = []
      X2 = []
      G1 = []
      G2 = []
      A1 = []
      A2 = []
      x1 = LAYER_IN1
      x2 = LAYER_IN2
      for i in range(DEPTH):
        x1, x2, g1, g2, a1, a2 = BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, x1, x2)
        X1.append(x1)
        X2.append(x2)
        G1.append(g1)
        G2.append(g2)
        A1.append(a1)
        A2.append(a2)

      return X1, X2, G1, G2, A1, A2

    input1 = Input(shape=(seq_size, dim), name="seq1")
    input2 = Input(shape=(seq_size, dim), name="seq2")
    


    X1 = dict()
    X2 = dict()
    G1 = dict()
    G2 = dict()
    A1 = dict()
    A2 = dict()

    for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
      X1[f'{i}'], X2[f'{i}'], G1[f'{i}'], G2[f'{i}'], A1[f'{i}'], A2[f'{i}'] = BlockCNN_single_deep(i, POOLING_SIZE, DEPTH, FILTERS, input1, input2)

    s1 = []
    s2 = []
    for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
      s1.extend(G1[f'{i}'])
      s2.extend(G2[f'{i}'])

    s1 = concatenate(s1)
    s2 = concatenate(s2)
    
    s1 = BatchNormalization(momentum=.9)(s1)
    s2 = BatchNormalization(momentum=.9)(s2)

    s1 = Dropout(DROPOUT1)(s1)
    s2 = Dropout(DROPOUT1)(s2)
    
    s1_shape = s1.shape[-1]
    DENSE1 = 744 
    d1 = []
    for i in range(DEPTH_DENSE1):
        d1.append(Dense(int(DENSE1*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))

    for i in range(DEPTH_DENSE1):
        s1 = d1[i](s1)
        s2 = d1[i](s2)
        s1 = Dropout(DROPOUT1)(s1)
        s2 = Dropout(DROPOUT1)(s2)
        
    s = concatenate([s1, s2])

    
    s_shape = s.shape[-1]
    DENSE2 = 328
        
    d2 = []
    for i in range(DEPTH_DENSE2):
        d2.append(Dense(int(DENSE2*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))

    for i in range(DEPTH_DENSE2):
        s = d2[i](s)
        s = Dropout(DROPOUT2)(s)

    output = Dense(1, activation='sigmoid')(s)
    model = Model(inputs=[input1, input2], outputs=[output])
    
    adam = Adam(learning_rate=learning_rate, amsgrad=True, epsilon=1e-6)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model