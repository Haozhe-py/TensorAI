from transformer import Transformer
import sys

model = Transformer(
    num_layers=4, d_model=128, num_heads=8, dff=512,
    input_vocab_size=1000, target_vocab_size=800,
    pe_input=10000, pe_target=10000
)                                                           #Create a Transformer Model

