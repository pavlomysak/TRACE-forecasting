class FCRBlock(nn.Module):
    def __init__(self, in_features, dropout, dilation):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        # feed-forward =========================================================
        self.feedforward = nn.Sequential(
            nn.LayerNorm(in_features), 
            nn.Linear(in_features, 2*in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*in_features, in_features)
        )

        # convolution =========================================================
        self.conv_norm = nn.LayerNorm(in_features)
        # depth-wise convolution
        self.dw = nn.Conv1d(in_features, in_features, 3, dilation = dilation, groups=in_features, padding='same')
        # point-wise convolution
        self.conv_relu = nn.ReLU()
        self.pw = nn.Conv1d(in_features, in_features, 1, padding='same')

        # RNN =========================================================
        self.rnn_norm = nn.LayerNorm(in_features)
        self.gru = nn.GRU(in_features, in_features, batch_first=True)


    def forward(self, x):

        y = self.feedforward(x)

        z = self.conv_norm(y + x)
        u = self.dw(z.transpose(2,1))
        u = self.pw(self.conv_relu(u))
        u = self.drop(u).transpose(2,1)

        v = self.rnn_norm(u + z)
        out, _ = self.gru(v)
        l = self.drop(out)

        return v + l



class TRACE(nn.Module):

    """
    Data: [Historical Target Variable, Numeric Covariates, Categorical Covariates, Date Data]
    Architecture Overview: [Feed-Forward -> Convolution -> LSTM]*k -> Self-Attention -> Maxout
    """

    def __init__(self, encoder_dim, num_blocks, dropout, horizon):
        super().__init__()

        self.horizon = horizon
        self.encoder_dim = encoder_dim
        
        # embedding =========================================================
        self.month_embedding = nn.Embedding(13, 7) # round(1.6*n_unique**0.56)
        self.week_embedding = nn.Embedding(53, 15)
        self.day_embedding = nn.Embedding(32, 11)
        self.sku_embedding = nn.Embedding(28, 10)
        self.family_embedding = nn.Embedding(19, 8)
        self.subcategory_embedding = nn.Embedding(6, 4)

        # [Embedding = 55 ; Target = 1; Other Numeric = 7] = 64
        in_features = 64
        
        # projection =========================================================
        self.front_proj = nn.Sequential(
            nn.LayerNorm(in_features), 
            nn.Linear(in_features, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # FCR Block =========================================================
        self.fcr_blocks = nn.ModuleList(
            [FCRBlock(encoder_dim, dropout = 0.2, dilation = int(i / 0.7)+2) for i in range(num_blocks)]
        )

        # attention =========================================================
        self.pre_attn_norm = nn.LayerNorm(encoder_dim)
        self.q_proj = nn.Linear(encoder_dim, encoder_dim)
        self.k_proj = nn.Linear(encoder_dim, encoder_dim)
        self.v_proj = nn.Linear(encoder_dim, encoder_dim)
        
        self.attention = nn.MultiheadAttention(
            encoder_dim, 
            num_heads = 4, 
            dropout = dropout, 
            batch_first = True
        )

        # additive attention pooling =========================================================
        self.attn_pool_w = nn.Linear(encoder_dim, encoder_dim)
        self.attn_pool_v = nn.Linear(encoder_dim, 1)

        # Classification Head =========================================================
        self.clf_pre_projection = nn.Linear(encoder_dim, encoder_dim*2)
        self.clf_post_projection = nn.Linear(encoder_dim, horizon)

        # Regression Head =========================================================
        self.rgr_pre_projection = nn.Linear(encoder_dim, encoder_dim*2)
        self.rgr_post_projection = nn.Linear(encoder_dim, horizon)

    def forward(self, historical_target, datetime_dt, categorical_dt, other_numeric_dt):

        # Embedding ==========================
        month_embed = self.month_embedding(datetime_dt[:,:,0].long())
        week_embed = self.week_embedding(datetime_dt[:,:,1].long())
        day_embed = self.day_embedding(datetime_dt[:,:,2].long())
        sku_embed = self.sku_embedding(categorical_dt[:,:,0].long())
        fam_embed = self.family_embedding(categorical_dt[:,:,1].long())
        subcat_embed = self.subcategory_embedding(categorical_dt[:,:,2].long())

        embedded_concat = torch.concat(
            (month_embed, week_embed, day_embed, sku_embed, fam_embed, subcat_embed), 
            dim=2).float()

        # Concatination and Projection ==========
        X = torch.concat(
            (historical_target.unsqueeze(-1).float(), embedded_concat, other_numeric_dt),
            dim=2).float()

        X = self.front_proj(X)
        
        # FCR Blocks ===========================
        for block in self.fcr_blocks:
            X = block(X)
        fcr_out = X

        # Self-Attention =======================
        X = self.pre_attn_norm(fcr_out)
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)
        attn_out, _ = self.attention(q, k, v)
        X = attn_out + fcr_out

        # Self-Attention Pooling ===============
        P = torch.tanh(self.attn_pool_w(X))
        scores = self.attn_pool_v(P).squeeze(-1)
        A = torch.softmax(scores, dim=1)
        Z = torch.einsum('bt,btd->bd', A, X)

        B = Z.shape[0]
        # Classification Head ==================
        clf_out = self.clf_pre_projection(Z).view(B, self.encoder_dim, 2).max(-1).values
        clf_out = self.clf_post_projection(clf_out)
        clf_out = torch.sigmoid(clf_out)

        # Regression Head ======================
        rgr_out = self.rgr_pre_projection(Z).view(B, self.encoder_dim, 2).max(-1).values
        rgr_out = self.rgr_post_projection(rgr_out)
        
        return clf_out, rgr_out
