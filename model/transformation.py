import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 16, use_cuda: str = 'cuda'):
        super().__init__()
        self.use_cuda = use_cuda
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size, device=use_cuda),
        ) 

        self.positions = nn.Parameter(torch.randn(hidden_size).to(use_cuda))


    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x += self.positions
        return x


class NeighAttention(nn.Module):
    def __init__(self, emb_size: int = 768, use_cuda: str = 'cuda'):
        super().__init__()
        self.W1 = nn.Linear(emb_size, emb_size, device=use_cuda)
        self.W2 = nn.Linear(2*emb_size, 2*emb_size, device=use_cuda)
        self.W3 = nn.Linear(2*emb_size, emb_size, device=use_cuda)


    def forward(self, x, x_neigh, ent):
        wneigh = self.W1(x_neigh)
        agg = torch.mean(wneigh, 1, True)
        cat = self.W2(torch.concat((agg, ent), 2))
        cat = F.softmax(cat, dim=-1)
        x_shape = cat.repeat(1,x.size()[-2],1)
        att_weigh = self.W3(x_shape)
        att_weigh = F.softmax(att_weigh, dim=-1)

        return att_weigh


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 4, dropout: float = 0, use_cuda: str = 'cuda'):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.q = nn.Linear(emb_size, emb_size, device=use_cuda) 
        self.k = nn.Linear(emb_size, emb_size, device=use_cuda) 
        self.v = nn.Linear(emb_size, emb_size, device=use_cuda) 
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size, device=use_cuda)

    def forward(self, q : Tensor, k : Tensor, v : Tensor, mask: Tensor = None, first: bool = True) -> Tensor:
        batch_size = q.shape[0]
        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)


        queries = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.head_dim ** (1/2)

        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, values) 
        out = out.permute(0, 2, 1, 3).contiguous()


        out = out.view(batch_size, -1, self.emb_size)
        out = self.projection(out)

        return out




class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0., use_cuda: str='cuda'):
        super().__init__()
        self.l1 = nn.Linear(emb_size, L * emb_size, device=use_cuda)
        self.l2 = nn.GELU()
        self.l3 = nn.Dropout(drop_p)
        self.l4 = nn.Linear(L * emb_size, emb_size, device=use_cuda)
    
    def forward(self, x):
        return self.l4(self.l3(self.l2(self.l1(x))))



class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
            forward_drop_p: float = 0., use_cuda: str = 'cuda',
                 **kwargs):
                 
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_size, device=use_cuda)
        self.attention = MultiHeadAttention(emb_size, use_cuda=use_cuda, **kwargs)
        self.drop1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size, device=use_cuda)
        self.ff = FeedForwardBlock(emb_size, L=forward_expansion, drop_p=forward_drop_p, use_cuda=use_cuda)
        self.drop2 = nn.Dropout(drop_p)

    def forward(self, x_in):
        
        x = self.norm1(x_in)
        x_att = self.drop1(self.attention(x, x, x)) 
        x_res1 = x_att + x_in
        x = self.norm2(x_res1)
        x_ff = self.drop2(self.ff(x))
        x_res2 = x_ff + x_res1
        
        return x_res2



class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    
    def forward(self, x_in):
        for layer in self.layers:
            x_in = layer(x_in)


        return x_in





class TransformerDecoderBlock(nn.Module):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
            forward_drop_p: float = 0., use_cuda: str = 'cuda', include_main_ent: bool = False,
                 **kwargs):
                 
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_size, device=use_cuda)
        self.attention1 = MultiHeadAttention(emb_size, use_cuda=use_cuda, **kwargs)
        self.drop1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size, device=use_cuda)
        self.graphsage = NeighAttention(emb_size, use_cuda)
        self.attention2 = MultiHeadAttention(emb_size, use_cuda=use_cuda, **kwargs)
        self.drop2 = nn.Dropout(drop_p)
        self.norm3 = nn.LayerNorm(emb_size, device=use_cuda)
        self.ff = FeedForwardBlock(emb_size, L=forward_expansion, drop_p=forward_drop_p, use_cuda=use_cuda)
        self.drop3 = nn.Dropout(drop_p)
        self.include_main_ent = include_main_ent

    def forward(self, x_graph, x_neigh, ent):
        
        x = self.norm1(x_graph)
        x_att = self.drop1(self.attention1(x, x, x)) 
        x_res1 = x_att + x_graph
        
        x = self.norm2(x_res1)
        neigh_att = self.graphsage(x, x_neigh, ent.unsqueeze(0))
        x_att = self.drop1(self.attention2(x, neigh_att, x, first=False)) 

        x_res2 = x_att + x_res1
        
        x = self.norm3(x_res2)
        x_ff = self.drop3(self.ff(x))
        x_res3 = x_ff + x_res2
        
        return x_res3



class TransformerDecoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(**kwargs) for _ in range(depth)])
    
    def forward(self, x_in, x_src, ent):
        for layer in self.layers:
            x_in = layer(x_in, x_src, ent)

        return x_in




class TransformationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, transformation_size: int = 1000, use_cuda: str = 'cuda'):
        super().__init__(
            nn.LayerNorm(emb_size, device=use_cuda),
            nn.Linear(emb_size, transformation_size, device=use_cuda))


class MLP(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size, use_cuda):
        super().__init__(
            nn.Linear(input_size, hidden_size,  device=use_cuda),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, device=use_cuda))


class ViT(nn.Module):
    def __init__(self,     
                input_size_n: int = 3,
                hidden_size_n: int = 3,
                input_size_g: int = 3,
                hidden_size_g: int = 3,
                depth: int = 12,
                transformation_size: int = 750,
                use_cuda: bool = True,
                include_main_ent = False,
                **kwargs):
        super().__init__()
        if use_cuda:
            self.use_cuda = 'cuda'
        else:
            
            self.use_cuda = 'cpu'
        self.include_main_ent = include_main_ent
        self.patch = PatchEmbedding(input_size_g, hidden_size_g, self.use_cuda)
        self.decode = TransformerDecoder(depth, emb_size=hidden_size_g, use_cuda=self.use_cuda, include_main_ent=self.include_main_ent, **kwargs)
        self.transform = TransformationHead(hidden_size_n, transformation_size, self.use_cuda)
        self.mlp_neigh = MLP(input_size_n, hidden_size_n, hidden_size_n, self.use_cuda)
        self.mlp_ent = MLP(input_size_n, hidden_size_n, hidden_size_n, self.use_cuda)

    def forward(self, x_neigh1, x_graph, ent):
       
        if x_neigh1.size()[1]==0:
            x_neigh1 = torch.ones(1,1,x_neigh1.size()[2], device=self.use_cuda)
        x_neigh = self.mlp_neigh(x_neigh1)
        ent = self.mlp_ent(ent)
        x_graph = self.patch(x_graph)
        
        x_trans = self.transform(self.decode(x_graph, x_neigh, ent))
        return x_trans

