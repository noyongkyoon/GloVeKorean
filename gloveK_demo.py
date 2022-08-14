from pathlib import Path
import os
import argparse
import pickle

import torch
import yaml
from gensim.models.keyedvectors import KeyedVectors
from glove import GloVe
import streamlit as st

def load_config():
    config_filepath = Path(__file__).absolute().parents[0] / "config.yaml"
    with config_filepath.open() as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config, key, value)
    return config

config = load_config()
with open(os.path.join(config.cooccurrence_dir, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

model = GloVe(
    vocab_size=config.vocab_size,
    embedding_size=config.embedding_size,
    x_max=config.x_max,
    alpha=config.alpha
)
tload = torch.load(config.output_filepath)
model.load_state_dict(tload)

keyed_vectors = KeyedVectors(vector_size=config.embedding_size)
keyed_vectors.add_vectors(
    keys=[vocab.get_token(index) for index in range(config.vocab_size)],
    weights=(model.weight.weight.detach()
        + model.weight_tilde.weight.detach()).numpy()
)

# title and description
st.write("""
# 한국어 GLoVe 체험
두 표현 사이의 유사도를 측정합니다. 각 표현은 여러 개의 낱말로 이루어질
수 있습니다.
""")
phrase1 = st.text_input("표현 1:", value="애완 동물", help="낱말들 사이에는 빈 칸 글자를 넣으세요.")
phrase2 = st.text_input("표현 2:", value="강아지 나 고양이", help="낱말들 사이에는 빈 칸 글자를 넣으세요.")
st.write('첫 번째 표현은 \"', phrase1, '\"이고 두 번째 표현은 \"', phrase2, '\"입니다.')
if st.button('유사도를 말 해 줘!'):
    li1 = []
    li2 = []
    for w in phrase1.split():
        try:
            dex = keyed_vectors.get_index(w, None)
            li1.append(w)
        except KeyError:
            print (f"\"{w}\"는 안알려진 낱말입니다.")
            print("낱말 연속체라면 미리 분절해서 넣으시지요.")
    for w in phrase2.split():
        try:
            keyed_vectors.get_index(w, None)
            li2.append(w)
        except KeyError:
            print (f"\"{w}\"는 안알려진 낱말입니다.")
            print("낱말 연속체라면 미리 분절해서 넣으시지요.")

    sim = keyed_vectors.n_similarity(li1, li2)
    st.write("{:.3f}".format(sim))
