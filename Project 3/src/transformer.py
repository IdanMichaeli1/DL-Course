import torch
import torch.nn as nn
import math





def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size % 2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0] 

    values, attention = None, None

    # Create window indices
    half_window = window_size // 2
    device = q.device
    
    if q.dim() == 3:
        q = q.unsqueeze(1)  # [Batch, 1, SeqLen, Dims]
        k = k.unsqueeze(1)  # [Batch, 1, SeqLen, Dims]
        v = v.unsqueeze(1)  # [Batch, 1, SeqLen, Dims]
    
    num_heads = q.shape[1]

    # Create window indices
    index = torch.arange(seq_len, device=device)
    window_indices = index[:, None] + torch.arange(-half_window, half_window + 1, device=device)
    window_indices = torch.clamp(window_indices, 0, seq_len - 1)
    
    # Expand window indices for batch and heads processing
    expanded_indices = window_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_heads, -1, -1, embed_dim)

    # Gather keys and values using the window indices
    k_flat = k.unsqueeze(3).expand(batch_size, num_heads, seq_len, seq_len, embed_dim)
    v_flat = v.unsqueeze(3).expand(batch_size, num_heads, seq_len, seq_len, embed_dim)
    k_gathered = torch.gather(k_flat, dim=3, index=expanded_indices)
    v_gathered = torch.gather(v_flat, dim=3, index=expanded_indices)

    # Compute attention scores
    q_expanded = q.unsqueeze(3)  # Shape: [Batch, Heads, SeqLen, 1, Dims]
    attention_scores = torch.matmul(q_expanded, k_gathered.transpose(-2, -1)).squeeze(3) / (embed_dim ** 0.5)
    
    if padding_mask is not None:
        padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(3).expand(attention_scores.size())
        attention_scores = attention_scores.masked_fill(padding_mask_expanded, float('-inf'))
    
    attention = torch.nn.functional.softmax(attention_scores, dim=-1)

    # Compute weighted sum
    values = torch.matmul(attention.unsqueeze(3), v_gathered).squeeze(3)

    return values, attention



class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''
        atten_output = self.self_attn(x, padding_mask)
        atten_output = self.dropout(atten_output)
        
        output = x + atten_output
        output = self.norm1(output)
        
        ff_output = self.feed_forward(output)
        ff_output = self.dropout(ff_output)
        
        output = output + ff_output
        output = self.norm2
        
        return output
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        embeded_sentece = self.encoder_embedding(sentence)
        embeded_sentece = self.positional_encoding(embeded_sentece)
        embeded_sentece = self.dropout(embeded_sentece)
        
        encoder_layers_output = self.encoder_layers(embeded_sentece, padding_mask)
        output = self.classification_mlp(encoder_layers_output)
        
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    