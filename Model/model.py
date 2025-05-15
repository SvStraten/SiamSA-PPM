
import os
import sys
path = os.getcwd()
sys.path.append(path)


import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.supports_masking = True

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=None, key_mask=mask)
        out1 = self.norm1(inputs + self.dropout(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + self.dropout(ffn_output, training=training))


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.supports_masking = True

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout, maxlen, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.supports_masking = True

        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.norm = layers.LayerNormalization()

    def call(self, x, training=False, mask=None):
        mask = self.embedding.compute_mask(x)  # shape: (batch_size, seq_len)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        return self.norm(x)


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_encoder(embed_dim, num_heads, ff_dim, num_layers, dropout, maxlen, vocab_size, hidden_dim, feature_dim):
    inputs = tf.keras.Input(shape=(maxlen,), dtype=tf.int32)
    
    x = TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        maxlen=maxlen,
        vocab_size=vocab_size
    )(inputs)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(hidden_dim, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(feature_dim, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    return tf.keras.Model(inputs, x, name="Encoder")



def get_predictor(feature_dim):
    inputs = tf.keras.Input(shape=(feature_dim,), dtype=tf.float32)

    x = layers.Dense(feature_dim // 4, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(feature_dim)(x)

    return tf.keras.Model(inputs, x, name="Predictor")



