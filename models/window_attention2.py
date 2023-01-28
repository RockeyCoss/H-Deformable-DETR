import torch
import torch.nn as nn


class WindowShiftAttention2(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 window_size=300,
                 shift_scheme='roll',
                 qkv_bias=True,
                 dropout=0.0):
        super(WindowShiftAttention2, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        head_embed_dims = embed_dim // num_heads
        self.scale = head_embed_dims ** -0.5

        self.qk_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        assert shift_scheme in ('roll', 'interleave')
        self.shift_scheme = shift_scheme

    def window_partition(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            (num_windows*B, window_size, C)
        """
        B, L, C = x.shape
        window_size = self.window_size
        x = x.view(B, L // window_size, window_size, C)
        x = x.reshape(-1, window_size, C)
        return x

    def windows_reverse(self, x, L):
        """
        Args:
            x: (num_windows*B, window_size, C)
        Returns:
            (B, L, C)
        """
        window_size = self.window_size
        B = int(x.size(0) / (L / window_size))
        x = x.view(B, L, x.size(-1))
        return x

    def forward(self,
                query,
                query_pos,
                shift_attn):
        """
        Args:
            query: b,q,c
            query_pos: b,q,c
            shift_attn: bool
        """
        B, L, C = query.shape
        assert L % self.window_size == 0

        if shift_attn and query.size(1) > self.window_size:
            one2many_q = query[:, self.window_size:]
            one2many_q_pos = query_pos[:, self.window_size:]
            B1, L1, C1 = one2many_q.shape
            if self.shift_scheme == 'roll':
                one2many_q = torch.roll(one2many_q,
                                        shifts=-(self.window_size // 2),
                                        dims=1)
                one2many_q_pos = torch.roll(one2many_q_pos,
                                            shifts=-(self.window_size // 2),
                                            dims=1)
            elif self.shift_scheme == 'interleave':
                one2many_q = one2many_q.reshape(B1,
                                                L1 // self.window_size,
                                                self.window_size,
                                                C1).transpose(1, 2).reshape(B1, L1, C1)
                one2many_q_pos = one2many_q_pos.reshape(B1,
                                                        L1 // self.window_size,
                                                        self.window_size,
                                                        C1).transpose(1, 2).reshape(B1, L1, C1)
            else:
                raise NotImplementedError()
            query = torch.cat((query[:, :self.window_size], one2many_q), dim=1)
            query_pos = torch.cat((query_pos[:, :self.window_size], one2many_q_pos), dim=1)

        # b*nW, window_size, c
        query = self.window_partition(query)
        w_query_pos = self.window_partition(query_pos)
        BW = query.size(0)
        # b*nW,num_heads,window_size,head_dim
        qk = self.qk_proj(query + w_query_pos).reshape(BW,
                                                       self.window_size,
                                                       2,
                                                       self.num_heads,
                                                       C // self.num_heads). \
            permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v_proj(query).reshape(BW,
                                       self.window_size,
                                       self.num_heads,
                                       C // self.num_heads). \
            permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        q = (attn @ v).transpose(1, 2).reshape(BW, self.window_size, C)
        q = self.proj(q)
        # B,L,C
        query = self.windows_reverse(q, L)

        if shift_attn and query.size(1) > self.window_size:
            one2many_q = query[:, self.window_size:]
            B1, L1, C1 = one2many_q.shape
            if self.shift_scheme == 'roll':
                one2many_q = torch.roll(one2many_q,
                                        shifts=self.window_size // 2,
                                        dims=1)
            elif self.shift_scheme == 'interleave':
                one2many_q = one2many_q.reshape(B1,
                                                self.window_size,
                                                L1 // self.window_size,
                                                C1).transpose(1, 2).reshape(B1, L1, C1)
            else:
                raise NotImplementedError()
            query = torch.cat((query[:, :self.window_size], one2many_q), dim=1)
        return query
