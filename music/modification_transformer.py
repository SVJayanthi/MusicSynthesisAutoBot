from fastai.basics import *
from fastai.text.models.transformer import TransformerXL
import numpy as np
import torch

class MusicTransformerXL(TransformerXL):
    "Exactly like fastai's TransformerXL, but with more aggressive attention mask: see `rand_window_mask`"
    def __init__(self, *args, encode_position=True, mask_steps=1, **kwargs):
        import inspect
        sig = inspect.signature(TransformerXL)
        arg_params = { k:kwargs[k] for k in sig.parameters if k in kwargs }
        super().__init__(*args, **arg_params)

        self.encode_position = encode_position
        if self.encode_position: self.beat_enc = BeatPositionEncoder(kwargs['d_model'])
            
        self.mask_steps=mask_steps
        
        
    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init: 
            self.reset()
            self.init = True

        benc = 0
        if self.encode_position:
            x,pos = x['x'], x['pos']
            benc = self.beat_enc(pos)

        bs,x_len = x.size()
        inp = self.drop_emb(self.encoder(x) + benc) #.mul_(self.d_model ** 0.5)
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        
        mask = self.rand_window_mask(x_len, m_len, inp.device, max_size=self.mask_steps, is_eval=not self.training) if self.mask else None
        if m_len == 0: mask[...,0,0] = 0
        #[None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]
    
    def rand_window_mask(x_len,m_len,device,max_size:int=None,p:float=0.2,is_eval:bool=False):
        if is_eval or np.random.rand() >= p or max_size is None:
            win_size,k = (1,1)
        else: win_size,k = (np.random.randint(0,max_size)+1,0)
        return (self.window_mask(x_len, device, m_len, size=(win_size,k)))
    
    def window_mask(x_len, device, m_len=0, size=(1,1)):
        win_size,k = size
        mem_mask = torch.zeros((x_len,m_len), device=device)
        tri_mask = torch.triu(torch.ones((x_len//win_size+1,x_len//win_size+1), device=device),diagonal=k)
        window_mask = tri_mask.repeat_interleave(win_size,dim=0).repeat_interleave(win_size,dim=1)[:x_len,:x_len]
        if x_len: window_mask[...,0] = 0 # Always allowing first index to see. Otherwise you'll get NaN loss
        mask = torch.cat((mem_mask, window_mask), dim=1)[None,None]
        return mask.bool() if hasattr(mask, 'bool') else mask.byte()
    
    
    


 # Beat encoder
class BeatPositionEncoder(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, emb_sz:int, beat_len=32, max_bar_len=1024):
        super().__init__()

        self.beat_len, self.max_bar_len = beat_len, max_bar_len
        self.beat_enc = nn.Embedding(beat_len, emb_sz, padding_idx=0)
        self.bar_enc = nn.Embedding(max_bar_len, emb_sz, padding_idx=0)
    
    def forward(self, pos):
        beat_enc = self.beat_enc(pos % self.beat_len)
        bar_pos = pos // self.beat_len % self.max_bar_len
        bar_pos[bar_pos >= self.max_bar_len] = self.max_bar_len - 1
        bar_enc = self.bar_enc((bar_pos))
        return beat_enc + bar_enc