import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Union
from dataclasses import dataclass, fields
from tensordict import TensorDict
from torch import Tensor
from entmax import entmax_bisect, entmax15, sparsemax #, normmax_bisect, budget_bisect

from utils.functions import batchify,gather_by_index,unbatchify,unbatchify_and_gather

def linear_layer(input_dim, output_dim, mean=0., std=1e-2, bias=True, bias_init_value=None):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim,bias=bias)
    nn.init.normal_(linear.weight, mean=mean, std=std)
    if bias and bias_init_value is not None:
        nn.init.constant_(linear.bias, bias_init_value)
    else:
        nn.init.zeros_(linear.bias)
    return linear

@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)

class PromptNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = 5 # C, O, TW, L, B, size
        self.model_params = args.model_params
        output_dim = args.model_params['embedding_dim']
        layer1 = nn.Linear(input_dim, output_dim,bias=False)
        nn.init.uniform_(layer1.weight)
        self.model = nn.Sequential(
            layer1,
            nn.LayerNorm(output_dim),
            linear_layer(output_dim,output_dim),
            nn.ReLU(),
            linear_layer(output_dim, output_dim//8), # task embedding
            nn.LayerNorm(output_dim//8),
            linear_layer(output_dim//8, output_dim),
        )

    def forward(self, td):
        return {"prompt": self.model(td['p_s_tag'][:,:5]).view(td.batch_size[0], 1, -1)}

class VRPModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = VRP_Encoder(**args.model_params)
        self.decoder = VRP_Decoder(**args.model_params)
        self.encoded_nodes = None  # (batch, problem+1, EMBEDDING_DIM)
        self.now_p_type = None
        self.prompt_net = PromptNet(args)

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert (not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
                    ), "infeasible action selected"
        return selected

    @staticmethod
    def sampling(logprobs, log, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1) #
        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert (not (~mask).gather(1, selected.unsqueeze(-1)).data.any()), "infeasible action selected"
        return selected

    def forward(self, td, env):
        #  td = TensorDict(td, batch_size=[td['locs'].shape[0]],device=td['locs'].device)
        args = self.args
        # init prompt out
        p_out = self.prompt_net(td)
        prompt = p_out['prompt']
        node_embed = self.encoder(td, prompt) 
        # multi_start node
        num_starts, action = env.select_start_nodes(td) # action repeat_num*bs   action num_starts*batch_size [1]*batch_size + [2]*batch_size + ...
        td = batchify(td, num_starts) # = td.expand(num_starts, batch_size).contiguous().view(batch_size * num_starts,)  td['locs'][0] == td['locs'][batch_size]
        logprobs_list = [torch.zeros_like(action, device=td.device)]
        actions_list = [action]
        # first node
        td.set("action", action)
        td = env.step(td)["next"]
        # set decoder k v
        decoder_k = reshape_by_heads(self.decoder.Wk(node_embed), head_num=args.model_params['head_num'])
        decoder_v = reshape_by_heads(self.decoder.Wv(node_embed), head_num=args.model_params['head_num']) # (batch, head_num, problem+1, qkv_dim)v
        decoder_single_head_k = node_embed.transpose(1, 2) # (batch, embedding, problem+1)
        cache = PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logprobs, mask = self.decoder(td, cache, num_starts)
            # select
            if self.training:
                select = VRPModel.sampling(logprobs, args.log, mask)
            else:
                select = VRPModel.greedy(logprobs,mask)
            logprobs = gather_by_index(logprobs, select, dim=1)
            td.set("action", select)
            actions_list.append(select)
            logprobs_list.append(logprobs)
            td = env.step(td)["next"]
            step += 1
        # post op
        logprobs = torch.stack(logprobs_list, 1)
        actions = torch.stack(actions_list, 1)
        td.set("reward", env.get_reward(td, actions)) #
        assert (logprobs > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
        log_likelihood_sum = logprobs.sum(1)  # Calculate log_likelihood [batch]
        outdict = {
            "reward": td["reward"], "log_likelihood": log_likelihood_sum, # bs * repeat_num
        }
        return outdict

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.embedding_depot = nn.Linear(3, embedding_dim) # locs, distance_limit, time_windows # TODO:
        self.embedding_node = nn.Linear(7, embedding_dim)  # 
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.layers2 = None
        #
        model_params_ = model_params.copy()
        model_params_['use_sparse'] = False
        self.layers2 = nn.ModuleList([EncoderLayer(**model_params_) for _ in range(encoder_layer_num)])
        self.layers1combine = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num)]) 
        self.layers2combine = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(encoder_layer_num-1)]) 


    def forward(self, td, prompt):
        """
        :param depot_xy: (batch, 1, 2)
        :param node_xy_demand: (batch, problem, 3)
        :return: out # (batch, problem+1, embedding)
        """
        # depot_feats = td["locs"][:, :1, :] # (batch, 1, 2)
        depot_feats = torch.cat(
            [
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
            ],
            -1,
        ) # (batch, 1, 4)
        node_feats = torch.cat(
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ), -1,) # (batch, N, 7)
        depot_feats = torch.nan_to_num(depot_feats, nan=0.0, posinf=0.0, neginf=0.0)
        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        bs, n, _7 = node_feats.shape
        global_embeddings = self.embedding_depot(depot_feats)  # [batch, 1, embed_dim]
        cust_embeddings = self.embedding_node(node_feats)  # [batch, N, embed_dim]
        
        out = torch.cat((global_embeddings, cust_embeddings), -2)  # [batch, N+1, embed_dim]
        out2 = out
      
      
        for i,layer in enumerate(self.layers):
            if i == 0 and prompt is not None:
                out2 = torch.cat((out2, prompt), dim=1) #[batch, N+2, embed_dim]
            out = layer(out) ################# layer 1 (sparse)
            if self.layers2 is not None:
                out2 = self.layers2[i](out2) ############# layer 2 (global)
                # combine global to sparse
                out = out + self.layers1combine[i](out2[:,:n+1])
                if i != len(self.layers)-1: 
                    # combine sparse to global
                    # add
                    out2_ = out2[:,:n+1] + self.layers2combine[i](out)
                    if prompt is not None: 
                        out2_ = torch.cat((out2_, out2[:,-1:]),dim=1)
                    out2 = out2_
        return out[:,:n+1] # (batch, problem+1, embedding)
    
class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        # MHA parameter
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.normalization_1 = Normalization(**model_params)
        # FFD parameter
        if model_params['ffd'] == 'ffd':
            self.feed_forward = FeedForward(**model_params)
        elif model_params['ffd'] == 'siglu':
            assert embedding_dim == 128
            self.feed_forward = ParallelGatedMLP()
        else:
            raise NotImplementedError
        self.normalization_2 = Normalization(**model_params)

    def multiheadattention(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num) # (batch, head_num, problem, qkv_dim)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num) # (batch, head_num, problem, qkv_dim)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num) # (batch, head_num, problem, qkv_dim)
        # prepare aentmax parameter
        out_concat = multi_head_attention(q, k, v, sparse=self.model_params['use_sparse']) # (batch, problem, head_num*qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)# (batch, problem, embedding)
        return multi_head_out

    def forward(self, x):
        # input1.shape: (batch, problem+1, embedding)
        h = self.normalization_1(x + self.multiheadattention(x))
        h = self.normalization_2(h + self.feed_forward(h))
        return h # shape: (batch, problem, embedding)
    
class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.Wq_last = nn.Linear(embedding_dim+5, head_num * qkv_dim, bias=False) # 
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, td, cache, num_starts, prompt=None):
        td = unbatchify(td, num_starts) #  num_starts * bs -> bs , num_starts
        # get last node
        cur_node_embedding = gather_by_index(cache.node_embeddings, td["current_node"])
        # get state original
        remaining_linehaul_capacity = td["vehicle_capacity"] - td["used_capacity_linehaul"]
        remaining_backhaul_capacity = td["vehicle_capacity"] - td["used_capacity_backhaul"]
        state_embedding = torch.cat([
            remaining_linehaul_capacity,remaining_backhaul_capacity,
            td["current_time"], td["current_route_length"], td["open_route"]
        ],-1) # bs, num start, 5
        # get context
        if prompt is None:
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1) # bs,num_locs,embed+5
        else:
            prompt = unbatchify(prompt, num_starts)
            prompt = rearrange(prompt, 'b n p e -> b n (p e)')
            context_embedding = torch.cat([cur_node_embedding, state_embedding, prompt], -1) # bs,num_locs,embed+5+...
        # get q
        glimpse_q = reshape_by_heads(self.Wq_last(context_embedding), head_num=self.model_params['head_num'])
        mask = td["action_mask"]
        # mha
        out_concat = multi_head_attention(glimpse_q, cache.glimpse_key, cache.glimpse_val, mask) # (batch, pomo, head_num*qkv_dim)
        mh_atten_out = self.multi_head_combine(out_concat) # (batch, pomo, embedding)
        # sha
        score = torch.matmul(mh_atten_out, cache.logit_key) # (batch, pomo, problem)
        score_scaled = score / self.model_params['sqrt_embedding_dim'] # (batch, pomo, problem)
        # post op
        logits = rearrange(score_scaled, "b s l -> (s b) l", s=num_starts)
        mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        logits = torch.tanh(logits) * self.model_params['logit_clipping']
        logits[~mask] = float("-inf")
        # logits = logits / temperature  # temperature scaling
        # probs = F.softmax(score_masked, dim=2)# (batch, pomo, problem)
        return F.log_softmax(logits, dim=-1),mask # Compute log probabilities


########################################
# NN SUB CLASS / FUNCTIONS
def reshape_by_heads(qkv, head_num):
    # q.(batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1) # (batch, n, head_num, key_dim)
    q_transposed = q_reshaped.transpose(1, 2) # (batch, head_num, n, key_dim)
    return q_transposed

def multi_head_attention(q, k, v, ninf_mask=None, sparse=False):
    # q (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.(batch, problem)
    # rank3_ninf_mask.(batch, group, problem)
    batch_s,head_num,n,key_dim = q.shape
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3)) # (batch, head_num, n, problem)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)
    if sparse == 'topk':
        out = 0 
        k_ = int(n/2)
        mask = torch.zeros(batch_s, head_num, n, n, device=score.device, requires_grad=False)
        mask.scatter_(-1, torch.topk(score_scaled, k=k_, dim=-1, largest=True)[1], 1.)
        attn = torch.where(mask > 0, score_scaled, torch.full_like(score_scaled, float('-inf')))
        attn = attn.softmax(dim=-1)
        out_ = (attn @ v)
        out = out + out_ 
        out_transposed = out.transpose(1, 2) # (batch, n, head_num, key_dim)
        out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim) # (batch, n, head_num*key_dim)
        return out_concat
    elif sparse is False:
        weights = nn.Softmax(dim=3)(score_scaled) # (batch, head_num, n, problem)
    elif sparse == 'relu':
        weights = nn.ReLU()(score_scaled)**2 # (batch, head_num, n, problem)
    elif sparse == 'entmax15':
        weights = entmax15(score_scaled, dim=3) # (batch, head_num, n, problem)
    elif sparse == 'sparsemax':
        weights = sparsemax(score_scaled, dim=3) # (batch, head_num, n, problem)
    else:
        raise NotImplementedError
    out = torch.matmul(weights, v) # (batch, head_num, n, key_dim)
    out_transposed = out.transpose(1, 2) # (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim) # (batch, n, head_num*key_dim)
    return out_concat

class Normalization(nn.Module): # post norm: first add, then norm
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_type = model_params['norm_type'] # instance or layer
        if self.norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif self.norm_type == 'layer': # layer
            self.norm = nn.LayerNorm(embedding_dim)
        elif self.norm_type == 'rms': # layer
            self.norm = RMSNorm(embedding_dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        # input: (batch, problem, embedding)
        if self.norm_type == 'instance':
            out = self.norm(x.transpose(1, 2)).transpose(1, 2) # (batch, problem, embedding)
        else:  # layer rms
            out = self.norm(x) # (batch, problem, embedding)
        return out

class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()
        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        self.multiple_of = multiple_of * model_parallel_size
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        ) # 512

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)
    
class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)
    def forward(self, input1):
        # input.(batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """
    def geglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)

class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



