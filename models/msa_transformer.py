import torch
import torch.nn as nn

import esm
from Bio import SeqIO
import string
from typing import List, Tuple
import os

os.environ['TORCH_HOME'] = '/work/jiaqi/Retrieval/esm_models'

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

if __name__ == '__main__':
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval().to('cuda:7')
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
    
    PDB_IDS = ["1a3a", "5ahw", "1xcr"]

    msas = {
            name: read_msa(f"esm/examples/data/{name.lower()}_1_A.a3m")[:10]
            for name in PDB_IDS
            }
    msa_transformer_predictions = {}
    msa_transformer_results = []
    for name, inputs in msas.items():
        # inputs = greedy_select(inputs, num_seqs=128) # can change this to pass more/fewer sequences
        msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
        msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
        # msa_transformer_predictions[name] = msa_transformer.predict_contacts(msa_transformer_batch_tokens)[0].cpu()
        msa_transformer_predictions[name] = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12])
        print(name, msa_transformer_predictions[name]['logits'].shape, msa_transformer_predictions[name]['representations'][12].shape)
        input()
        # metrics = {"id": name, "model": "MSA Transformer (Unsupervised)"}
        # metrics.update(evaluate_prediction(msa_transformer_predictions[name], contacts[name]))
        # msa_transformer_results.append(metrics)
    # msa_transformer_results = pd.DataFrame(msa_transformer_results)
    # display(msa_transformer_results)