import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
from tqdm import tqdm
import argparse
import copy
import csv
from datetime import datetime




parser = argparse.ArgumentParser()

parser.add_argument(
    "--video_dir",
    type=str,
    required=True,
    help="Path to video latent directory"
)

parser.add_argument(
    "--text_dir",
    type=str,
    required=True,
    help="Path to text embedding directory"
)

args = parser.parse_args()

VIDEO_LATENT_DIR = args.video_dir

TEXT_LATENT_DIR = args.text_dir



LOG_FILE = "training_log.csv"

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "epoch",
            "avg_loss"
        ])






class VideoPatchEmbed(nn.Module):

    def __init__(
        self,
        in_channels=16,
        patch_size=2,
        embed_dim=512
    ):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Linear(
            in_channels * patch_size * patch_size,
            embed_dim
        )

    def forward(self, x):

        """
        x:
        (B,T,C,H,W)

        Example:
        (B,10,16,28,37)
        """

        B,T,C,H,W = x.shape

        p = self.patch_size

        # crop width to divisible size
        H_new = (H // p) * p
        W_new = (W // p) * p

        x = x[:,:,:,:H_new,:W_new]

        # patchify
        x = x.reshape(
            B,
            T,
            C,
            H_new // p,
            p,
            W_new // p,
            p
        )

        # rearrange
        x = x.permute(
            0,1,3,5,2,4,6
        )

        """
        becomes:

        (B,T,H_p,W_p,C,p,p)
        """

        # flatten patches
        x = x.reshape(
            B,
            T,
            H_new // p,
            W_new // p,
            C * p * p
        )

        # flatten tokens
        x = x.reshape(
            B,
            -1,
            C * p * p
        )

        # linear projection
        x = self.proj(x)

        return x




class TimestepEmbedding(nn.Module):

    def __init__(
        self,
        dim=512
    ):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):

        """
        t:
        (B,)
        """

        half = self.dim // 2

        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(
                0,
                half,
                device=t.device
            ).float()
            / half
        )

        args = t[:,None].float() * freqs[None]

        emb = torch.cat([
            torch.sin(args),
            torch.cos(args)
        ], dim=-1)

        emb = self.mlp(emb)

        return emb





def get_1d_sincos(pos, dim):

    omega = torch.arange(
        dim // 2,
        dtype=torch.float32,
        device=pos.device
    )

    omega /= dim / 2

    omega = 1.0 / (10000 ** omega)

    out = pos[:,None] * omega[None,:]

    sin = torch.sin(out)
    cos = torch.cos(out)

    return torch.cat([sin, cos], dim=1)




class VideoPosEmbedding(nn.Module):

    def __init__(
        self,
        T=12,
        H=14,
        W=18,
        dim=512
    ):
        super().__init__()

        self.T = T
        self.H = H
        self.W = W
        self.dim = dim

        d_t = dim // 4
        d_h = dim // 4
        d_w = dim - d_t - d_h

        t_pos = torch.arange(T).float()
        h_pos = torch.arange(H).float()
        w_pos = torch.arange(W).float()

        t_emb = get_1d_sincos(
            t_pos,
            d_t
        )

        h_emb = get_1d_sincos(
            h_pos,
            d_h
        )

        w_emb = get_1d_sincos(
            w_pos,
            d_w
        )

        pos = []

        for t in range(T):
            for h in range(H):
                for w in range(W):

                    emb = torch.cat([
                        t_emb[t],
                        h_emb[h],
                        w_emb[w]
                    ])

                    pos.append(emb)

        pos = torch.stack(pos)

        self.register_buffer(
            "pos_embedding",
            pos.unsqueeze(0)
        )

    def forward(self, x):

        """
        x:
        (B,N,D)
        """

        N = x.shape[1]

        return x + self.pos_embedding[:, :N]


        


VIDEO_LATENT_DIR = "/kaggle/input/datasets/nisav007/text-to-video-datasets/videolatent/videolatent/content/videolatent"

# list files
files = sorted(os.listdir(VIDEO_LATENT_DIR))

print("Total files:", len(files))

print(files[:5])






def normalize_temporal(
    latent,
    target_frames=12
):

    T = latent.shape[1]

    # -------------------
    # random crop
    # -------------------

    if T > target_frames:

        start = random.randint(
            0,
            T - target_frames
        )

        latent = latent[
            :,
            start:start+target_frames
        ]

    # -------------------
    # pad
    # -------------------

    elif T < target_frames:

        pad_frames = target_frames - T

        last_frame = latent[:, -1:]

        pad = last_frame.repeat(
            1,
            pad_frames,
            1,
            1,
            1
        )

        latent = torch.cat(
            [latent, pad],
            dim=1
        )

    return latent




class Attention(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8
    ):
        super().__init__()

        self.heads = heads

        self.head_dim = dim // heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

    def forward(self, x):

        B,N,D = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q,k,v = qkv

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.1 if self.training else 0.0
        )

        out = out.transpose(
            1,
            2
        )

        out = out.reshape(
            B,
            N,
            D
        )

        out = self.proj(out)

        return out

class CrossAttention(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8
    ):
        super().__init__()

        self.heads = heads

        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)

        self.kv = nn.Linear(dim, dim * 2)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x,
        context
    ):

        B,N,D = x.shape

        M = context.shape[1]

        q = self.q(x)

        kv = self.kv(context)

        q = q.reshape(
            B,N,self.heads,self.head_dim
        ).transpose(1,2)

        kv = kv.reshape(
            B,M,2,self.heads,self.head_dim
        )

        kv = kv.permute(
            2,0,3,1,4
        )

        k,v = kv

        out = F.scaled_dot_product_attention(
            q,
            k,
            v
        )

        out = out.transpose(
            1,
            2
        ).reshape(B,N,D)

        out = self.proj(out)

        return out



class AdaLN(nn.Module):

    def __init__(self, dim=512):

        super().__init__()

        self.act = nn.SiLU()

        self.proj = nn.Linear(
            dim,
            dim * 3
        )

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x,
        t_embed
    ):

        scale, shift, gate = self.proj(
            self.act(t_embed)
        ).chunk(
            3,
            dim=-1
        )

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)

        x = x * (1 + scale) + shift

        return x, gate



class SwiGLU(nn.Module):

    def forward(self, x):

        x, gate = x.chunk(2, dim=-1)

        return x * F.silu(gate)
    




class FeatureCaches:

    def __init__(

        self,

        

        max_cache_size=4
    ):

        

        self.max_cache_size = max_cache_size

        self.spatial_cache = {}

    # =====================================================
    # APPEND FEATURES
    # =====================================================

    def append_spatial(

        self,

        layer_idx,

        features
    ):

        # ---------------------------------------------
        # detach + cpu
        # ---------------------------------------------

        features = features.detach()

        lst = self.spatial_cache.get(
            layer_idx,
            []
        )

        lst.append(features)

        # ---------------------------------------------
        # keep recent cache only
        # ---------------------------------------------

        if len(lst) > self.max_cache_size:

            lst.pop(0)

        self.spatial_cache[layer_idx] = lst

    # =====================================================
    # RETRIEVE CACHE
    # =====================================================

    def get_spatial_mean(

        self,

        layer_idx
    ):

        lst = self.spatial_cache.get(
            layer_idx,
            []
        )

        if len(lst) == 0:

            return None

        stacked = torch.stack(
            lst,
            dim=0
        )

        out = stacked.mean(dim=0)

        return out

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):

        self.spatial_cache.clear()







class DiTBlock(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8,
        mlp_ratio=4,
        use_cross=True,
        use_spatial_cache=True,
        cache_stride=2,
        dropout=0.1,

    ):
        super().__init__()
       

        self.use_cross = use_cross
        self.use_spatial_cache = use_spatial_cache
        self.cache_stride = cache_stride

        # =================================================
        # NORMALIZATION
        # =================================================
        



        self.norm1 = nn.RMSNorm(
            dim,
            elementwise_affine=False
        )

        self.norm2 = nn.RMSNorm(
            dim,
            elementwise_affine=False
        )

        self.norm3 = nn.RMSNorm(
            dim,
            elementwise_affine=False
        )

        # =================================================
        # ADALN
        # =================================================

        self.adaln1 = AdaLN(dim)
        self.adaln2 = AdaLN(dim)
        self.adaln3 = AdaLN(dim)

        # =================================================
        # ATTENTION
        # =================================================

        self.attn = Attention(
            dim,
            heads
        )

        if use_cross:

            self.cross_attn = CrossAttention(
                dim,
                heads
            )

        # =================================================
        # DROPOUT
        # =================================================

        self.dropout = nn.Dropout(dropout)

        # =================================================
        # LAYERSCALE
        # =================================================

        self.gamma_attn = nn.Parameter(
            1e-4 * torch.ones(dim)
        )

        self.gamma_mlp = nn.Parameter(
            1e-4 * torch.ones(dim)
        )

        if use_cross:

            self.gamma_cross = nn.Parameter(
                1e-4 * torch.ones(dim)
            )

        # =================================================
        # CACHE
        # =================================================

        if use_spatial_cache:

            self.cache_alpha = nn.Parameter(
                torch.tensor(-2.0)
            )

            self.cache_proj = nn.Linear(
                dim,
                dim
            )

            self.summary_proj = nn.Linear(
                dim,
                dim
            )





        # =================================================
        # MLP
        # =================================================

        hidden_dim = dim * mlp_ratio

        self.mlp = nn.Sequential(

            nn.Linear(
                dim,
                hidden_dim * 2
            ),

            SwiGLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                dim
            )
        )



    def forward(
        self,
        x,
        text_tokens,
        t_embed,
        cache=None,
        
        layer_idx=None
    ):

        # =================================================
        # SELF ATTENTION
        # =================================================

        h = self.norm1(x)

        h, gate_attn = self.adaln1(
            h,
            t_embed
        )

        x = x + (
            gate_attn* 
            self.gamma_attn
            *
            self.dropout(
                self.attn(h)
            )
        )


        # =================================================
        # SPATIAL CACHE
        # =================================================

        if (
            self.use_spatial_cache
            and
            cache is not None
        ):

            prev_idx = (
                layer_idx
                -
                self.cache_stride
            )

            if prev_idx >= 0:

                spatial_feat = cache.get_spatial_mean(
                    prev_idx
                )

                if spatial_feat is not None:

                    spatial_feat = self.cache_proj(
                        spatial_feat
                    )

                    x = x + (
                        torch.sigmoid(
                            self.cache_alpha
                        )
                        *
                        spatial_feat
                    )

            # ---------------------------------------------
            # Projected semantic summary
            # ---------------------------------------------

            summary = x.mean(
                dim=1,
                keepdim=True
            )

            summary = self.summary_proj(
                summary
            )

            cache.append_spatial(
                layer_idx,
                summary
            )

        # =================================================
        # CROSS ATTENTION
        # =================================================

        if self.use_cross:

            h = self.norm2(x)

            h, gate_cross = self.adaln2(
                h,
                t_embed
            )

            x = x + (gate_cross* 
                self.gamma_cross
                *
                self.dropout(
                    self.cross_attn(
                        h,
                        text_tokens
                    )
                )
            )

        # =================================================
        # MLP
        # =================================================

        h = self.norm3(x)

        h , gate_mlp= self.adaln3(
            h,
            t_embed
        )

        x = x + (gate_mlp*
            self.gamma_mlp
            *
            self.dropout(
                self.mlp(h)
            )
        )

        return x


class DiT(nn.Module):

    def __init__(

        self,

        depth=6,

        dim=512,

        heads=8,

        use_spatial_cache=True,

        cache_stride=2
    ):

        super().__init__()

        self.depth = depth

        self.use_spatial_cache = use_spatial_cache

        # =================================================
        # TRANSFORMER BLOCKS
        # =================================================

        self.blocks = nn.ModuleList()

        for i in range(depth):

            # ---------------------------------------------
            # use cross attention every alternate block
            # ---------------------------------------------

            use_cross = (
                i % 2 == 0
            )

            block = DiTBlock(

                dim=dim,

                heads=heads,

                use_cross=use_cross,

                use_spatial_cache=use_spatial_cache,

                cache_stride=cache_stride
            )

            self.blocks.append(block)

        # =================================================
        # FINAL NORMALIZATION
        # =================================================

        self.norm = nn.LayerNorm(dim)

        # =================================================
        # OUTPUT HEAD
        # =================================================

        self.head = nn.Linear(
            dim,
            dim
        )

        # =================================================
        # STABLE DIFFUSION INIT
        # =================================================

        nn.init.zeros_(
            self.head.weight
        )

        nn.init.zeros_(
            self.head.bias
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(

        self,

        x,

        text_tokens,

        t_embed
    ):

        """
        x:
        (B,N,D)

        Example:
        (1,3024,512)
        """

        # =================================================
        # CREATE FRESH CACHE PER VIDEO
        # =================================================

        cache = None

        

        if self.use_spatial_cache:

            cache = FeatureCaches()
              

            cache.clear()

        # =================================================
        # TRANSFORMER BLOCKS
        # =================================================

        for idx, block in enumerate(self.blocks):

            x = block(

                x,

                text_tokens,

                t_embed,

                cache=cache,

                

                layer_idx=idx
            )
                    # =================================================
        # FINAL NORMALIZATION
        # =================================================

        x = self.norm(x)

        # =================================================
        # PREDICT NOISE
        # =================================================

        x = self.head(x)

        return x





class DiffusionScheduler(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):
        super().__init__()

        self.timesteps = timesteps

        # -------------------
        # beta schedule
        # -------------------

        betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=torch.float32
        )

        alphas = 1.0 - betas

        alpha_bars = torch.cumprod(
            alphas,
            dim=0
        )

        # -------------------
        # precompute sqrt terms
        # -------------------

        sqrt_alpha_bars = torch.sqrt(
            alpha_bars
        )

        sqrt_one_minus_alpha_bars = torch.sqrt(
            1 - alpha_bars
        )

        # -------------------
        # register buffers
        # -------------------

        self.register_buffer(
            "betas",
            betas
        )

        self.register_buffer(
            "alphas",
            alphas
        )

        self.register_buffer(
            "alpha_bars",
            alpha_bars
        )

        self.register_buffer(
            "sqrt_alpha_bars",
            sqrt_alpha_bars
        )

        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            sqrt_one_minus_alpha_bars
        )

    def add_noise(
        self,
        x0,
        t
    ):

        """
        x0:
        (B,T,C,H,W)

        t:
        (B,)
        """

        noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[
            t
        ].view(
            -1,
            1,
            1,
            1,
            1
        )

        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[
                t
            ].view(
                -1,
                1,
                1,
                1,
                1
            )
        )

        xt = (

            sqrt_alpha_bar * x0

            +

            sqrt_one_minus_alpha_bar * noise
        )

        return xt, noise



class VideoUnpatchify(nn.Module):

    def __init__(
        self,
        dim=512,
        out_channels=16,
        patch_size=(1,2,2),
        T=12,
        H=28,
        W=36
    ):
        super().__init__()

        self.pt, self.ph, self.pw = patch_size

        self.T = T
        self.H = H
        self.W = W

        self.out_channels = out_channels

        # -------------------
        # patch vector size
        # -------------------

        patch_dim = (
            out_channels
            * self.pt
            * self.ph
            * self.pw
        )

        # -------------------
        # reverse projection
        # -------------------

        self.proj = nn.Linear(
            dim,
            patch_dim
        )

    def forward(self, x):

        """
        x:
        (B,3024,512)
        """

        B,N,D = x.shape

        # -------------------
        # reverse projection
        # -------------------

        x = self.proj(x)

        # -------------------
        # reshape patches
        # -------------------

        x = x.reshape(

            B,

            self.T,

            self.H // self.ph,

            self.W // self.pw,

            self.out_channels,

            self.pt,

            self.ph,

            self.pw
        )

        # -------------------
        # rearrange
        # -------------------

        x = x.permute(
            0,
            1,
            4,
            2,
            6,
            3,
            7,
            5
        )

        # -------------------
        # merge dimensions
        # -------------------

        x = x.reshape(

            B,

            self.T,

            self.out_channels,

            self.H,

            (self.W // self.pw) * self.pw
        )

        return x



def diffusion_training_step(

    latent,
    text_tokens,

    scheduler,
    timestep_embedder,

    patcher,
    posembed,

    model,
    unpatchify
):

    """
    latent:
    (B,12,16,28,37)

    text_tokens:
    (B,77,512)
    """

    device = latent.device

    # -------------------
    # bf16
    # -------------------

    latent = latent.to(torch.bfloat16)

    text_tokens = text_tokens.to(
        torch.bfloat16
    )

    B = latent.shape[0]

    # -------------------
    # sample timestep
    # -------------------

    t = torch.randint(
        0,
        scheduler.timesteps,
        (B,),
        device=device
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        # -------------------
        # add diffusion noise
        # -------------------

        xt, noise = scheduler.add_noise(
            latent,
            t
        )

        # -------------------
        # geometry alignment
        # patcher internally crops
        # 37 -> 36
        # -------------------

        xt = xt[:,:,:,:28,:36]

        noise = noise[:,:,:,:28,:36]

        # -------------------
        # patchify
        # -------------------

        tokens = patcher(xt)

        # -------------------
        # positional embedding
        # -------------------

        tokens = posembed(tokens)

        # -------------------
        # timestep embedding
        # -------------------

        t_embed = timestep_embedder(t)

        t_embed = t_embed.to(
            tokens.dtype
        )

        # -------------------
        # DiT predicts noise
        # -------------------

        pred_tokens = model(

            tokens,

            text_tokens,

            t_embed
        )

        # -------------------
        # reconstruct latent noise
        # -------------------

        pred_noise = unpatchify(
            pred_tokens
        )

        # -------------------
        # diffusion loss
        # -------------------

        loss = F.mse_loss(
            pred_noise,
            noise
        )

    return loss



model = DiT(

depth=6,

dim=512,

heads=8

).cuda()


ema_model = copy.deepcopy(
    model
)

ema_model.eval()

for p in ema_model.parameters():

    p.requires_grad = False



@torch.no_grad()
def update_ema(

    model,

    ema_model,

    decay=0.999
):

    model_state = model.state_dict()

    ema_state = ema_model.state_dict()

    for key in ema_state.keys():

        if not torch.is_floating_point(
            ema_state[key]
        ):
            continue

        ema_state[key].mul_(decay)

        ema_state[key].add_(

            model_state[key],

            alpha=1.0 - decay
        )




scheduler = DiffusionScheduler().cuda()


patcher = VideoPatchEmbed(

in_channels=16,

patch_size=2,

embed_dim=512

).cuda()


posembed = VideoPosEmbedding(

T=12,

H=14,

W=18,

dim=512

).cuda()


unpatchify = VideoUnpatchify(

dim=512,

out_channels=16,

patch_size=(1,2,2),

T=12,

H=28,

W=36

).cuda()


timestep_embedder = TimestepEmbedding(
dim=512
).cuda()

optimizer = torch.optim.AdamW(

model.parameters(),

lr=1e-4,

betas=(0.9,0.95),

weight_decay=1e-2
)





EPOCHS = 10

model.train()




for epoch in range(EPOCHS):

    total_loss = 0

    pbar = tqdm(
        range(len(files))
    )

 
    for idx in pbar:
        

    # -------------------
    # video latent path
    # -------------------

        file=files[idx]

        latent_path = os.path.join(
            VIDEO_LATENT_DIR,
            file
        )

        # -------------------
        # load latent
        # -------------------

        latent = torch.load(
            latent_path
        )

        latent = normalize_temporal(
            latent,
            12
        )

        # -------------------
        # corresponding text file
        # -------------------

        text_file = file.replace(
            ".pt",
            "_text.pt"
        )

        text_path = os.path.join(
            TEXT_LATENT_DIR,
            text_file
        )

        # -------------------
        # load text embeddings
        # -------------------

        text_embeddings = torch.load(
            text_path
        )

        # -------------------
        # save sample
        # -------------------

        dataset={

            "file": file,

            "latent": latent,

            "text_embeddings": text_embeddings
        }


        sample = dataset

        latent = sample[
            "latent"
        ].cuda()

        CFG_DROPOUT = 0.15

        text_tokens = random.choice(
            sample["text_embeddings"]
        ).cuda()

        # Classifier-Free Guidance training
        if random.random() < CFG_DROPOUT:

            text_tokens = torch.zeros_like(
                text_tokens
            )
            
    	

        # -------------------
        # zero grad
        # -------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------
        # diffusion step
        # -------------------

        loss = diffusion_training_step(

            latent,
            text_tokens,

            scheduler,
            timestep_embedder,

            patcher,
            posembed,

            model,
            unpatchify
        )

        # -------------------
        # NaN check
        # -------------------

        if torch.isnan(loss):

            print(
                "NaN loss detected"
            )

            break

        # -------------------
        # backward
        # -------------------

        loss.backward()

        # -------------------
        # gradient clipping
        # -------------------

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0
        )

        # -------------------
        # optimizer step
        # -------------------

        optimizer.step()


        update_ema(

            model,

            ema_model,

            decay=0.999
        )


        total_loss += loss.item()

        pbar.set_description(

            f"Epoch {epoch+1} "
            f"Loss {loss.item():.4f}"
        )

    avg_loss = (
        total_loss
        / len(files)
    )



    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            epoch + 1,
            avg_loss
        ])



    print(
        f"\nEpoch {epoch+1}"
    )

    print(
        f"Average Loss: "
        f"{avg_loss:.4f}"
    )


        
        
    # -------------------
    # save checkpoint
    # -------------------





    torch.save(

        model.state_dict(),

        f"dit_S_epoch_{epoch+1}.pth"
    )

    torch.save(

        ema_model.state_dict(),

        f"dit_S_epoch_{epoch+1}_ema.pth"
    )
  

    torch.cuda.empty_cache()
