import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
import pdb
from typing import Tuple, Union
from model.CLIP_model import CLIP_visual_encoder,LayerNorm,Transformer,Transformer_decoder,QuickGELU
from collections import OrderedDict
import math

class text_generator_Trans(nn.Module):
    def __init__( self, 
                 embed_dim : int,
                 context_length: int,
                 vocab_size: int,
                 sos_token: int,
                 eos_token: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 droprate: float):

        super().__init__()

        self.context_length = context_length
        self.transformer_dec = Transformer_decoder(
            width=embed_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask_enc=self.generate_square_mask(),
            attn_mask_dec=None
        )


        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = None
        self.and_token = None
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim,padding_idx=0)

        scale = embed_dim ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.context_length, embed_dim))
        self.ln_final = LayerNorm(embed_dim)
        if droprate > 0:
            self.dropout = nn.Dropout(p=droprate, inplace=True)
        else:
            self.dropout = nn.Identity()

        self.text_projection = nn.Parameter(scale * torch.randn(embed_dim,  vocab_size))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.embed_dim ** -0.5)
    def generate_square_mask(self):
        mask = (torch.triu(torch.ones((self.context_length, self.context_length))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, fea_emb, tokens_encoded,debug=False):
        #print("#### START caption_generator ...")
        #print(tokens_encoded.shape)
        x = self.token_embedding(tokens_encoded) # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        fea_emb = fea_emb.permute(1, 0, 2)  # NLD -> LND

        x,k,v,z = self.transformer_dec(x,fea_emb,fea_emb)
        text_emb_Ndim = z.permute(1, 0, 2) #torch.Size([BS, 65, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)
        x = self.dropout(x)
        x = x @ self.text_projection #torch.Size([64, 65, 10509])

        ## Get text embedding at the eos token
        for cnt,tokens in enumerate(tokens_encoded):
            pos_eos = torch.where(tokens == self.eos_token)[0].tolist()
            if len(pos_eos) == 0:
                pos_eos = tokens.size(0)-1
            else:
                pos_eos = pos_eos[0] #Get the position of the first eos token
            if cnt == 0:
                text_emb_1dim = text_emb_Ndim[cnt:cnt+1,pos_eos:pos_eos+1] #torch.Size([1, 1, 512])
            else:
                text_emb_1dim = torch.cat([text_emb_1dim,text_emb_Ndim[cnt:cnt+1,pos_eos:pos_eos+1]],dim=0) #torch.Size([BS, 1, 512])
        return x,text_emb_1dim


    def pred_next_token(self, fea_emb, pred_prev_tokens_emb,debug=False):
        #print("#### START predicting the next token ...")
        x = pred_prev_tokens_emb # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        fea_emb = fea_emb.permute(1, 0, 2)  # NLD -> LND
        
        x,k,v,z = self.transformer_dec(x,fea_emb,fea_emb)
        text_emb_Ndim = z.permute(1, 0, 2) #torch.Size([BS, 65, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)
        x = self.dropout(x)
        x = x @ self.text_projection #torch.Size([64, 65, 10509])

        return x,text_emb_Ndim


    def gen_text(self,fea_emb,beam_size,temperature,prefix=None,unk_token=None,and_token=None,drop_unk=True,drop_bigrams=True):
        """
        #### Adapted from the source code of ArtEmis dataset paper
        """  
        if unk_token:
            self.unk_token = unk_token
        if and_token:
            self.and_token = and_token

        batch_size,_,fea_d = fea_emb.size()
        assert batch_size == 1
        batch_size = beam_size
        fea_emb = fea_emb.repeat(beam_size,1,1)
        device = fea_emb.get_device()
        # CHECK HERE
        """
        if device == -1:
            device = 'cpu'
        """
        top_k_scores = torch.zeros(beam_size, 1).to(device)
        complete_seqs = list()
        complete_text_emb_1dim = list()
        complete_seqs_scores = list()

        ## initialize with sos
        pred_prev_tokens_emb = torch.zeros((batch_size,self.context_length,self.embed_dim)).to(device)
        pred_captions = torch.zeros((batch_size,self.context_length)).to(device).long()
        pred_captions[:,0] = torch.Tensor([self.sos_token]*batch_size).to(device).long()
        k_prev_words = torch.LongTensor([[self.sos_token]] * batch_size).to(device)  # (k, 1)

        pred_prev_tokens_emb[:,0] = self.token_embedding(pred_captions)[:,0] 
        pred_captions_prob = torch.Tensor()
        k = beam_size

        cnt_start = 1

        ## Initialize with prefix excluding eos if exist
        if prefix != None:
            assert len(prefix) == 1 #only work for batch size = 1
            prefix = prefix[0]
            pos_eos = torch.where(prefix == self.eos_token)[0].tolist()
            if len(pos_eos) == 0:
                pos_eos = prefix.size(0)-1
            else:
                pos_eos = pos_eos[0] #Get the position of the first eos token
            pred_captions[:,:pos_eos] = torch.Tensor([prefix[:pos_eos].tolist()]*k).to(device).long()
            pred_prev_tokens_emb[:,:pos_eos] = self.token_embedding(pred_captions)[:,:pos_eos] 
                
            # Tensor to store top k previous words at each step;
            k_prev_words = torch.LongTensor([[prefix[pos_eos-1]]] * k).to(device)  # (k, 1)
            cnt_start = pos_eos
        for cnt in range(cnt_start,self.context_length):
            #print(pred_prev_tokens_emb[0,0:5,0:5])
            pred_next_token_prob,text_emb_Ndim = self.pred_next_token(fea_emb,pred_prev_tokens_emb)
            pred_next_token_prob = pred_next_token_prob[:,cnt-1:cnt,:]

            prob_max,indice = torch.max(pred_next_token_prob,dim=-1)

            scores = pred_next_token_prob[:,0,:]  # (s, vocab_size)
            if temperature != 1:
                scores /= temperature
            scores = F.log_softmax(scores, dim=1)

            if drop_bigrams and cnt > 2:
                # drop bi-grams with frequency higher than 1. (x y ... x y)
                prev_usage = pred_captions[:, :cnt-1]
                x, y = torch.where(prev_usage == k_prev_words)
                y += 1 # word-after-last-in-prev-usage
                y = pred_captions[x, y]
                #print("x",x)
                #print("y",y)
                scores[x,y] = -math.inf
        
            if drop_unk:
                scores[:, self.unk_token] = -math.inf

            # drop x x (repeated words)
            x = range(pred_captions.size(0))
            y = k_prev_words[:,0]
            scores[x,y] = -math.inf

            ## drop x and x
            if cnt > 2: 
                x, y = torch.where(k_prev_words == self.and_token)
                pre_and_word = pred_captions[x, cnt-2]
                #print("x",x)
                #print("pre_and_word",k_prev_words)
                scores[x, pre_and_word] = -math.inf
                pre_and_word = pred_captions[x, cnt-3]
                #print("x",x)
                #print("pre_and_word",k_prev_words)
                scores[x, pre_and_word] = -math.inf
                pre_and_word = pred_captions[x, cnt-4]
                #print("x",x)
                #print("pre_and_word",k_prev_words)
                scores[x, pre_and_word] = -math.inf
            
            ## Only 1 and
            if cnt > 2:
                ## drop x and x
                #pdb.set_trace()
                x, y = torch.where(pred_captions == self.and_token)
                scores[x,and_token] = -math.inf
    
            scores = top_k_scores.expand_as(scores) + scores
            if cnt == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            #top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            #pdb.set_trace()
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            prev_word_inds = prev_word_inds.long()
            next_word_inds = top_k_words % self.vocab_size
            pred_captions = pred_captions[prev_word_inds]
            pred_captions[:,cnt] = next_word_inds
            

            # Find completed sequences 
            incomplete_inds = [ind for ind, word in enumerate(next_word_inds) if word != self.eos_token]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Add completed sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(pred_captions[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
                text_emb_Ndim = text_emb_Ndim[complete_inds]
                text_emb_1dim = text_emb_Ndim[:,cnt:cnt+1]
                complete_text_emb_1dim.extend(text_emb_1dim.tolist())
            k -= len(complete_inds)  # reduce beam length accordingly

            if k == 0:
                break

            if cnt == self.context_length-1 and len(complete_seqs_scores) == 0:# Do not have any completed sequence, but reach the context length
                complete_inds = incomplete_inds
                complete_seqs.extend(pred_captions[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
                text_emb_Ndim = text_emb_Ndim[complete_inds]
                text_emb_1dim = text_emb_Ndim[:,cnt:cnt+1]
                complete_text_emb_1dim.extend(text_emb_1dim.tolist())
            
            pred_captions = pred_captions[incomplete_inds]
            pred_prev_tokens_emb = pred_prev_tokens_emb[incomplete_inds]
            fea_emb = fea_emb[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1) 
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            pred_prev_tokens_emb[:,:cnt+1] = self.token_embedding(pred_captions)[:,:cnt+1]


        s_idx = np.argsort(complete_seqs_scores)[::-1]
        complete_seqs_scores = [complete_seqs_scores[i] for i in s_idx]
        complete_seqs = [complete_seqs[i] for i in s_idx]
        complete_text_emb_1dim = [complete_text_emb_1dim[i] for i in s_idx]
        complete_seqs_scores = torch.Tensor(complete_seqs_scores).to(device) #torch.Size([3])
        complete_seqs = torch.Tensor(complete_seqs).to(device) #torch.Size([3, 65])
        complete_text_emb_1dim = torch.Tensor(complete_text_emb_1dim).to(device) #torch.Size([3, 1, 512])
        return complete_seqs_scores,complete_seqs,complete_text_emb_1dim


