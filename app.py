"""
You need to run the app from the root
To run the app
$ streamlit run app.py
"""

import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

import model, sample, encoder

import streamlit as st
from PIL import Image

st.title('Automatic Text Generation')

with st.sidebar:
    st.subheader('Welcome!!!!')
   
    
    st.text("Reference")
    st.write("[Beginner’s Guide to Retrain GPT-2 (117M) to Generate Custom Text Content](https://medium.com/ai-innovation/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f)")

    
    st.subheader('Members:')
    st.text("Zahra Hatami")
    st.text("Van Tien Nguyen")
    st.text("Christina-Zoi Mavroeidi")
    st.text("Joshua Paul Marion Joseph")
    st.text("David Raphael Bravo Marcial")



def generate_paper(
    key_words,
    model_name='paper',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    ):
    
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(key_words)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                
    return text


key_words = st.text_input('Input your key words:')

if st.button('Generate'):
    if not key_words:
        st.warning('You need to enter key words')
        st.stop()
        
    paper = generate_paper(key_words)
    st.markdown(paper)
