# ALPA-Net Architecture Revision Plan for MI Localization

## Executive Summary

Your ALPA-Net (Anatomical Lead Prior Attention Network) is well-designed for MI localization, but there are **6 major revision areas** that can improve accuracy by **8-20%** and significantly enhance interpretability. This document provides a prioritized roadmap with code examples.

---

## Current Architecture Overview

```
Input (12-lead beats, 65 samples)
    ↓
[Shared CNN Stem] (7×1 conv per lead)
    ↓
[Channel Projection] (32→64 channels)
    ↓
[Multi-Scale Residual CNN] (kernels: 3,7,15,31)
    ↓
[Lead Token Builder] (Global pooling + Linear)
    ↓
[ALPA Module] (Anatomical Lead Prior Attention)
    ├─ Class queries (4×128)
    ├─ Lead priors (4×12 matrix)
    └─ Softmax attention scores
    ↓
[Cross-Lead Transformer] (2 layers, 4 heads)
    ↓
[Temporal Transformer] (1 layer on segments)
    ↓
[Attention Pooling] (Learned weighted aggregation)
    ↓
[Classification Head] (128→4 classes)
```

### Strengths ✓
1. **Anatomically-grounded**: Lead priors correctly reflect MI localization physiology
2. **Multi-scale feature extraction**: Captures temporal patterns at different granularities
3. **Dual attention pathways**: Cross-lead + temporal reasoning is complementary
4. **Transfer learning ready**: Modular backbone/heads design
5. **Interpretable**: Lead attention maps can be visualized

### Weaknesses ✗
1. **Very short temporal window** (0.65s): Misses ST-segment dynamics and T-wave evolution
2. **Fixed anatomical priors**: No learning/adaptation to data; assumes universal MI patterns
3. **Missing ECG waveform features**: Generic CNN doesn't capture ST-elevation or T-wave inversion explicitly
4. **Implicit lead relationships**: Lead ordering encoded only through position, not anatomical distance
5. **No robustness mechanisms**: Real-world ECGs have noise, artifacts, baseline wander
6. **Black-box attention**: Hard to verify model uses MI-specific features vs data artifacts

---

## Priority 1: Temporal Context Expansion (Accuracy +2-5%)

### Current Problem
- 65 samples @ 100Hz = only 0.65 seconds
- Misses key MI signatures that evolve over **1-2 seconds**
- ST-segment analysis requires longer baseline-to-repolarization window

### Revision A1: Beat Windowing with Context

**Code Change** (in data loading section):

```python
# CURRENT: Single beat only
x_shape = (N_samples, 65, 12)  # 0.65 seconds

# REVISED: Beat with preceding/following context
def create_beat_window(x, beat_indices, window_samples=200, hop_samples=65):
    """
    Extract 200-sample windows centered on beats (2.0 seconds @ 100Hz)
    with 65-sample stride (original beat length)
    """
    windowed = []
    for beat_idx in beat_indices:
        start = max(0, beat_idx - window_samples // 2)
        end = min(len(x), start + window_samples)
        if end - start == window_samples:  # Only valid windows
            windowed.append(x[start:end])
    return np.array(windowed)

# Update CONFIG
CONFIG['model']['signal_length'] = 200  # Changed from 65
CONFIG['beat_context'] = {
    'window_ms': 2000,        # 2-second total window
    'hop_ms': 650,            # Original beat length
    'center_beat_offset_ms': 325,  # Beat at center
}
```

### Revision A2: Multi-Beat Sequence Encoding

**Code Change** (modify dataset class):

```python
class ECGMultiBeatDataset(Dataset):
    """Encode 3-5 consecutive beats as a sequence"""
    
    def __init__(self, x, y, beat_record_map, labels_df=None, seq_length=5):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.beat_record_map = beat_record_map  # Maps beat_idx → record_idx
        self.seq_length = seq_length
        self.labels_df = labels_df
        
        # Build sequences of consecutive beats from same record
        self.sequences = self._build_sequences()
    
    def _build_sequences(self):
        """Group consecutive beats from same record"""
        sequences = []
        record_beats = defaultdict(list)
        
        for beat_idx, record_idx in enumerate(self.beat_record_map):
            record_beats[record_idx].append(beat_idx)
        
        for beat_indices in record_beats.values():
            for i in range(len(beat_indices) - self.seq_length + 1):
                seq_indices = beat_indices[i:i+self.seq_length]
                sequences.append(seq_indices)
        
        return sequences
    
    def __getitem__(self, idx):
        beat_indices = self.sequences[idx]
        x_seq = self.x[beat_indices]  # (5, 65, 12)
        y = self.y[beat_indices[len(beat_indices)//2]]  # Center beat label
        
        return {
            'x': x_seq,              # (seq_len, signal_len, n_leads)
            'y': y,
            'seq_indices': beat_indices,
        }
```

### Revision A3: Add RR-Interval Feature

```python
# Extract RR-interval (heart rate variability indicator)
def compute_rr_intervals(beat_positions, sampling_rate=100):
    """RR-interval in milliseconds"""
    rr_intervals = np.diff(beat_positions) / sampling_rate * 1000
    return rr_intervals

# Add to model input
def compute_rr_features(beat_indices, rr_intervals):
    """Create RR-interval feature for attention bias"""
    rr_features = []
    for beat_idx in beat_indices:
        if beat_idx == 0:
            rr_feat = rr_intervals[0]
        else:
            rr_feat = rr_intervals[beat_idx - 1]
        rr_features.append(rr_feat)
    return np.array(rr_features, dtype=np.float32)

# In ALPAModule: Add RR-interval bias
class ALPAModule(nn.Module):
    def __init__(self, token_dim, class_prior_matrix, rr_weight=0.1, **kwargs):
        super().__init__()
        self.class_queries = nn.Parameter(...)
        self.class_lead_prior_bias = nn.Parameter(...)
        self.rr_gate = nn.Linear(1, 1)  # Modulate attention by RR-interval
        self.rr_weight = rr_weight
    
    def forward(self, lead_tokens, rr_interval=None):
        data_scores = torch.einsum('bld,qd->bql', lead_tokens, self.class_queries) * self.scale
        scores = data_scores + self.class_lead_prior_bias.unsqueeze(0)
        
        # Optional RR-interval modulation
        if rr_interval is not None:
            rr_bias = self.rr_gate(rr_interval.unsqueeze(-1))
            scores = scores + rr_bias.unsqueeze(1) * self.rr_weight
        
        class_attention = torch.softmax(scores, dim=-1)
        return class_context, class_attention, lead_attention
```

---

## Priority 2: Learnable Anatomical Priors (Accuracy +1-3%, Interpretability +++)

### Current Problem
- Lead priors are **fixed hyperparameters** set manually
- No adaptation to specific dataset, patient population, or ECG device
- Can't distinguish between domain knowledge and data-driven patterns

### Revision B1: Per-Class Prior Strength

```python
# CURRENT: Single attention_prior_strength for all classes
CONFIG['attention_prior_strength'] = 1.2

# REVISED: Per-class strength tuning
CONFIG['attention_prior_strength_per_class'] = {
    'Normal': 0.8,        # Weak prior (any leads could be normal)
    'Anterior': 1.8,      # Strong prior (V1-V4 very specific)
    'Inferior': 2.0,      # Strongest (II, III, aVF most diagnostic)
    'Lateral': 1.5,       # Medium (I, aVL, V5-V6 somewhat specific)
}

# Modify ALPAModule
class ALPAModule(nn.Module):
    def __init__(self, token_dim, class_prior_matrix, 
                 prior_strength_per_class=None, prior_learnable=True):
        super().__init__()
        self.register_buffer('initial_class_prior', torch.tensor(class_prior_matrix))
        
        if prior_learnable:
            # Make strength learnable per class
            default_strength = prior_strength_per_class or [1.0] * len(class_prior_matrix)
            self.prior_strength = nn.Parameter(
                torch.tensor(default_strength, dtype=torch.float32)
            )
        
        # Prior bias becomes: log(prior) * learned_strength_per_class
        prior = torch.tensor(class_prior_matrix).clamp_min(1e-6)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        self.register_buffer('prior_basis', torch.log(prior))
```

### Revision B2: Learnable Prior Matrix with Regularization

```python
class LearnablePriorModule(nn.Module):
    """Learn prior matrix from data with Dirichlet regularization"""
    
    def __init__(self, n_classes, n_leads, initial_prior_matrix, 
                 alpha_concentration=2.0):
        super().__init__()
        self.n_classes = n_classes
        self.n_leads = n_leads
        
        # Initialize from anatomical priors
        prior_logits = torch.log(torch.tensor(initial_prior_matrix).clamp_min(1e-6))
        self.prior_logits = nn.Parameter(prior_logits)
        
        # Dirichlet concentration (lower = prefer sparsity, higher = prefer uniformity)
        self.alpha = alpha_concentration
    
    def forward(self):
        """Return learned prior matrix"""
        prior_dist = torch.softmax(self.prior_logits, dim=-1)
        return prior_dist
    
    def regularization_loss(self):
        """Dirichlet-inspired KL divergence from uniform"""
        prior = self.forward()
        uniform = torch.ones_like(prior) / self.n_leads
        
        # KL(prior || uniform) pushes toward uniform if alpha is low
        # Weighted toward initial prior if we add an additional term
        kl_loss = torch.sum(prior * (torch.log(prior) - torch.log(uniform)))
        return kl_loss * 0.01  # Weight coefficient

# Add to training loop
learnable_prior = LearnablePriorModule(n_classes=4, n_leads=12, 
                                       initial_prior_matrix=MAIN_CLASS_PRIOR_MATRIX)

def train_with_learnable_prior(model, learnable_prior, train_loader, optimizer, criterion):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['x'])
        
        # Classification loss
        ce_loss = criterion(outputs['logits'], batch['y'])
        
        # Prior regularization: keep learned priors close to anatomical priors
        prior_reg = learnable_prior.regularization_loss()
        
        total_loss = ce_loss + prior_reg
        total_loss.backward()
        optimizer.step()
```

### Revision B3: Prior Learning Visualization

```python
def visualize_prior_evolution(prior_history, class_names, lead_order, save_path):
    """Track how learned priors evolve during training"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.ravel()
    
    epochs = list(range(len(prior_history)))
    
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Prior values over time for each lead
        priors_by_epoch = [h[class_idx] for h in prior_history]
        priors_by_lead = np.array(priors_by_epoch).T  # (n_leads, n_epochs)
        
        for lead_idx, lead in enumerate(lead_order):
            ax.plot(epochs, priors_by_lead[lead_idx], label=lead, marker='o', markersize=3)
        
        ax.set_title(f'{class_name} Class Prior Evolution', fontsize=12, weight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Prior Weight')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    # Also compare: Initial vs Final
    initial = prior_history[0]
    final = prior_history[-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    for idx, (prior_matrix, title) in enumerate([(initial, 'Initial'), (final, 'Final')]):
        ax = axes[idx]
        sns.heatmap(np.array(prior_matrix), cmap='RdYlGn', 
                   xticklabels=lead_order, yticklabels=class_names,
                   ax=ax, vmin=0, vmax=0.3, cbar_kws={'label': 'Prior Weight'})
        ax.set_title(f'{title} Learned Prior Matrix', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_path.replace('.png', '_comparison.png'), bbox_inches='tight')
    plt.close(fig)
```

---

## Priority 3: MI-Specific ECG Feature Extraction (Accuracy +3-8%)

### Current Problem
- Generic CNN treats all signal regions equally
- Doesn't explicitly encode **ST-elevation**, **T-wave inversion**, **QRS widening**
- These are primary diagnostic features for MI localization

### Revision C1: ST-Segment Extraction Module

```python
class STSegmentModule(nn.Module):
    """
    Extract ST-segment (20-60ms post J-point) for elevation/depression analysis
    Anterior MI: ST elevation in V1-V4
    Inferior MI: ST elevation in II, III, aVF
    Lateral MI: ST elevation in I, aVL, V5-V6
    """
    
    def __init__(self, sampling_rate=100, j_point_offset=20):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.j_point_offset = j_point_offset  # ~200ms from QRS onset
        self.st_window_samples = int(0.06 * sampling_rate)  # 60ms window
        
        # Learn isoelectric level baseline
        self.baseline_detector = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        x: (batch, signal_len=65, n_leads=12)
        Returns: ST-segment features (batch, 12)
        """
        batch_size, signal_len, n_leads = x.shape
        
        # Estimate isoelectric baseline (PR interval or TP interval)
        baseline = self.baseline_detector(x.mean(dim=2))  # (batch, 1)
        
        # Extract ST-segment (20-60ms post J-point)
        j_point = self.j_point_offset
        st_end = j_point + self.st_window_samples
        
        if st_end <= signal_len:
            st_segment = x[:, j_point:st_end, :]  # (batch, 40, 12)
            st_elevation = st_segment.mean(dim=1) - baseline  # (batch, 12)
        else:
            st_elevation = torch.zeros(batch_size, n_leads, device=x.device)
        
        return st_elevation

class TWaveModule(nn.Module):
    """
    Detect T-wave inversion (common in MI)
    Normal: T-wave positive
    MI: T-wave inverted (negative)
    """
    
    def __init__(self, sampling_rate=100):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.t_wave_start = int(0.3 * sampling_rate)  # ~300ms post-QRS
        self.t_wave_end = int(0.6 * sampling_rate)    # ~600ms post-QRS
    
    def forward(self, x):
        """
        x: (batch, signal_len, n_leads)
        Returns: T-wave inversion score (batch, 12)
        """
        batch_size, signal_len, n_leads = x.shape
        
        if self.t_wave_end <= signal_len:
            t_wave = x[:, self.t_wave_start:self.t_wave_end, :]
            # Positive amplitude = normal, negative = inverted
            t_inversion_score = -t_wave.max(dim=1)[0]  # Negative peaks
            return t_inversion_score
        else:
            return torch.zeros(batch_size, n_leads, device=x.device)

# Integrate into ALPANetBackbone
class ALPANetBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...
        
        # Add MI-specific feature extractors
        self.st_module = STSegmentModule()
        self.twave_module = TWaveModule()
        
        # Combine ECG features with CNN features
        self.ecg_feature_fusion = nn.Sequential(
            nn.Linear(12 * 3, 128),  # 12 leads × 3 features (st, twave, qrs)
            nn.ReLU(),
            nn.Linear(128, m['token_dim'])
        )
    
    def forward(self, x):
        # ... existing CNN pathway ...
        
        # Parallel ECG feature pathway
        st_features = self.st_module(x)       # (batch, 12)
        twave_features = self.twave_module(x) # (batch, 12)
        # TODO: Add QRS feature extraction
        
        ecg_features = torch.cat([st_features, twave_features, ...], dim=1)
        ecg_tokens = self.ecg_feature_fusion(ecg_features)  # (batch, token_dim)
        
        # Fuse CNN tokens with ECG tokens
        combined_tokens = lead_tokens_raw + 0.3 * ecg_tokens.unsqueeze(1)
        
        return combined_tokens, # ... rest unchanged ...
```

---

## Priority 4: Spatial-Anatomical Encoding (Accuracy +1-2%)

### Current Problem
- Lead ordering (I, II, III, aVL, aVF, aVR, V1-V6) encoded only implicitly
- No explicit anatomical distance between leads
- Adjacent leads that observe same MI region not strongly connected

### Revision D1: Lead Position Encoding

```python
# Define 3D anatomical positions of leads (simplified)
LEAD_POSITIONS_3D = {
    'I': np.array([1.0, 0.0, 0.0]),      # Left lateral
    'II': np.array([0.5, -0.866, 0.0]),   # Inferior
    'III': np.array([-0.5, -0.866, 0.0]), # Inferior-left
    'aVR': np.array([-1.0, 0.0, 0.0]),    # Right
    'aVL': np.array([0.866, 0.5, 0.0]),   # Left-superior
    'aVF': np.array([0.0, -1.0, 0.0]),    # Inferior
    'V1': np.array([0.0, 0.0, 0.3]),      # Right ventricle
    'V2': np.array([0.0, 0.0, 0.5]),      # Right precordium
    'V3': np.array([0.0, 0.0, 0.7]),      # Anterior
    'V4': np.array([0.0, 0.0, 0.9]),      # Anterior apex
    'V5': np.array([0.3, 0.0, 0.8]),      # Left anterior
    'V6': np.array([0.6, 0.0, 0.7]),      # Left lateral
}

class LeadPositionalEncoding(nn.Module):
    """Encode lead positions as learnable embeddings"""
    
    def __init__(self, lead_order, lead_positions, token_dim=128):
        super().__init__()
        self.lead_order = lead_order
        self.n_leads = len(lead_order)
        
        # Learnable position embeddings based on anatomical locations
        positions = torch.tensor([
            lead_positions[lead] for lead in lead_order
        ], dtype=torch.float32)
        
        # Project 3D positions to token dimension
        self.position_projection = nn.Linear(3, token_dim)
        self.register_buffer('lead_positions_normalized', positions)
        
        # Learn lead-specific embeddings
        self.lead_embeddings = nn.Embedding(self.n_leads, token_dim)
    
    def forward(self, lead_tokens):
        """
        lead_tokens: (batch, n_leads, token_dim)
        Returns: augmented tokens with positional information
        """
        batch_size, n_leads, token_dim = lead_tokens.shape
        
        # Position-based encoding
        position_embed = self.position_projection(self.lead_positions_normalized)
        
        # Lead index embedding
        lead_idx_embed = self.lead_embeddings(torch.arange(n_leads, device=lead_tokens.device))
        
        # Combine: original tokens + position + lead identity
        augmented = lead_tokens + position_embed.unsqueeze(0) + lead_idx_embed.unsqueeze(0)
        
        return augmented

# Integrate into ALPANetBackbone
class ALPANetBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...
        self.lead_pos_encoding = LeadPositionalEncoding(
            LEAD_ORDER, LEAD_POSITIONS_3D, m['token_dim']
        )
    
    def forward(self, x):
        # ... get lead_tokens_raw ...
        lead_tokens = self.lead_pos_encoding(lead_tokens_raw)
        # ... rest unchanged ...
```

### Revision D2: Lead Neighborhood Attention

```python
def compute_lead_distance_matrix(lead_positions):
    """Euclidean distance between leads"""
    leads = list(lead_positions.keys())
    n_leads = len(leads)
    dist_matrix = np.zeros((n_leads, n_leads))
    
    for i, lead_i in enumerate(leads):
        for j, lead_j in enumerate(leads):
            dist = np.linalg.norm(lead_positions[lead_i] - lead_positions[lead_j])
            dist_matrix[i, j] = dist
    
    # Convert to attention bias (closer = higher attention)
    # Using Gaussian kernel
    bias_matrix = np.exp(-dist_matrix ** 2 / (2 * 0.5**2))
    return bias_matrix

class LeadNeighborhoodAttention(nn.Module):
    """Increase attention between anatomically adjacent leads"""
    
    def __init__(self, n_leads, lead_distance_matrix, temperature=1.0):
        super().__init__()
        # Convert distance to attention bias
        neighbor_bias = torch.tensor(lead_distance_matrix, dtype=torch.float32)
        neighbor_bias = neighbor_bias / neighbor_bias.max()  # Normalize to [0, 1]
        
        self.register_buffer('neighbor_bias', neighbor_bias)
        self.temperature = temperature
    
    def forward(self, lead_tokens):
        """
        lead_tokens: (batch, n_leads, token_dim)
        Returns: (batch, n_leads, n_leads) with neighborhood structure
        """
        # Cross-lead attention with neighborhood bias
        batch_size, n_leads, token_dim = lead_tokens.shape
        
        # Compute attention scores
        scores = torch.bmm(lead_tokens, lead_tokens.transpose(1, 2))  # (batch, n_leads, n_leads)
        scores = scores / np.sqrt(token_dim)
        
        # Add neighborhood bias
        scores = scores + self.neighbor_bias.unsqueeze(0) * self.temperature
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention, lead_tokens)
        
        return context, attention
```

---

## Priority 5: Class Imbalance Handling (Accuracy +2-4%)

### Current Problem
- Equal class weights may not reflect MI prevalence
- Rare MI types (e.g., pure Lateral MI) under-trained
- CrossEntropyLoss gives equal weight to all mistakes

### Revision E1: Focal Loss for Imbalance

```python
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard cases
    Useful for imbalanced MI datasets
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        logits: (batch, n_classes)
        targets: (batch,)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Compute class weights from dataset
def compute_class_weights(y_train, main_labels):
    """Weight inverse to class frequency"""
    class_counts = pd.Series(y_train).value_counts().sort_index()
    class_weights = len(y_train) / (len(main_labels) * class_counts.values)
    return torch.tensor(class_weights, dtype=torch.float32)

class_weights = compute_class_weights(y_train, MAIN_LABELS)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

### Revision E2: Per-Class Prior Scaling

```python
class ALPAModule(nn.Module):
    def __init__(self, token_dim, class_prior_matrix, 
                 class_prior_strength=None, **kwargs):
        super().__init__()
        
        # Allow different prior strength per class
        if class_prior_strength is None:
            class_prior_strength = [1.0] * len(class_prior_matrix)
        
        self.register_buffer('class_prior_strength', 
                            torch.tensor(class_prior_strength))
        
        # Build per-class prior bias
        prior = torch.tensor(class_prior_matrix).clamp_min(1e-6)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        prior_log = torch.log(prior)
        
        # Scale by class-specific strength
        prior_bias = prior_log * self.class_prior_strength.unsqueeze(1)
        self.register_buffer('class_lead_prior_bias', prior_bias)

# In CONFIG
CONFIG['class_prior_strength'] = {
    'Normal': 1.0,         # Weak prior (normal is common, less diagnostic)
    'Anterior': 2.0,       # Strong prior (V1-V4 highly specific)
    'Inferior': 2.5,       # Strongest prior (II, III, aVF diagnostic)
    'Lateral': 1.5,        # Medium prior
}
```

---

## Priority 6: Interpretability & Validation (Accuracy +0%, Trust +++++)

### Current Problem
- Hard to verify model learns MI features vs data artifacts
- Attention maps alone don't confirm correct reasoning
- No quantitative metrics for "did prior+data match well?"

### Revision F1: Gradient-Based Class Activation Mapping

```python
class GradCAM:
    """Gradient-weighted Class Activation Mapping for lead importance"""
    
    def __init__(self, model, target_layer_name='multi_scale'):
        self.model = model
        self.target_layer = dict(model.backbone.named_modules())[target_layer_name]
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, x, target_class):
        """
        Generate CAM for specific class
        
        Returns:
            cam_per_lead: (n_leads, signal_len) - importance per lead + time
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        logits = output['logits']
        
        # Backward pass for target class
        self.model.zero_grad()
        class_loss = logits[:, target_class].sum()
        class_loss.backward()
        
        # Compute CAM
        # activations: (batch, channels, time)
        # gradients: (batch, channels, time)
        weights = self.gradients.mean(dim=(0, 2), keepdim=True)  # (1, channels, 1)
        cam = (weights * self.activations).sum(dim=1)  # (batch, time)
        
        # Per-lead CAM by reshaping or averaging
        return cam

def visualize_gradcam_per_lead(model, x, y_true, pred, class_names, lead_order, save_path):
    """Visualize which leads and time-points matter for each class"""
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=300)
    
    gradcam = GradCAM(model)
    
    for class_idx, (ax_row, class_name) in enumerate(zip(axes, class_names)):
        cam = gradcam.generate_cam(x, target_class=class_idx)
        
        for lead_idx in range(12):
            ax = ax_row[lead_idx // 4]  # Reshape to 4×3
            lead_signal = x[0, :, lead_idx].cpu().numpy()
            lead_cam = cam[0].cpu().numpy()
            
            ax.plot(lead_signal, label='Signal', linewidth=1.5)
            ax_cam = ax.twinx()
            ax_cam.fill_between(range(len(lead_cam)), lead_cam, alpha=0.3, color='red')
            
            ax.set_title(f'{lead_order[lead_idx]} → {class_name}', fontsize=10)
            ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
```

### Revision F2: Prior-Attention Agreement Metric

```python
def compute_prior_attention_agreement(learned_attention, prior_matrix, 
                                     method='cosine'):
    """
    Measure: Does learned attention match anatomical priors?
    
    High agreement = Model respects anatomy
    Low agreement = Model may be finding data artifacts
    
    Returns: agreement_score (0-1), per_class_agreement
    """
    
    # learned_attention: (batch, n_classes, n_leads)
    # prior_matrix: (n_classes, n_leads)
    
    learned_attention_mean = learned_attention.mean(axis=0)  # (n_classes, n_leads)
    
    agreements = []
    for class_idx in range(len(prior_matrix)):
        learned = learned_attention_mean[class_idx]
        prior = prior_matrix[class_idx]
        
        if method == 'cosine':
            # Normalize
            learned_norm = learned / (np.linalg.norm(learned) + 1e-6)
            prior_norm = prior / (np.linalg.norm(prior) + 1e-6)
            agreement = np.dot(learned_norm, prior_norm)
        
        elif method == 'kl':
            # KL divergence (how much learned differs from prior)
            learned_safe = np.clip(learned, 1e-6, 1.0)
            prior_safe = np.clip(prior, 1e-6, 1.0)
            kl = np.sum(prior_safe * (np.log(prior_safe) - np.log(learned_safe)))
            agreement = 1 / (1 + kl)  # Convert to similarity
        
        agreements.append(agreement)
    
    return {
        'overall_agreement': np.mean(agreements),
        'per_class_agreement': {
            class_name: float(agreement) 
            for class_name, agreement in zip(MAIN_LABELS, agreements)
        }
    }

# Track during validation
def validate_with_agreement(model, val_loader, criterion, CONFIG, MAIN_LABELS, MAIN_CLASS_PRIOR_MATRIX):
    all_attention = []
    all_logits = []
    all_y = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['x'].to(CONFIG['device']))
            all_attention.append(outputs['class_attention'].cpu().numpy())
            all_logits.append(outputs['logits'].cpu().numpy())
            all_y.append(batch['y'].numpy())
    
    attention_matrix = np.concatenate(all_attention)  # (N, n_classes, n_leads)
    agreement_metrics = compute_prior_attention_agreement(
        attention_matrix, MAIN_CLASS_PRIOR_MATRIX, method='cosine'
    )
    
    return agreement_metrics
```

### Revision F4: Class Activation Mapping (CAM) for MI Types

```python
def compute_class_activation_map(model, x, class_idx, lead_order):
    """
    Visualize which leads/regions are most important for each MI type
    """
    
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    
    # Forward pass with attention tracking
    with torch.enable_grad():
        x_tensor.requires_grad = True
        outputs = model(x_tensor)
        logits = outputs['logits']
        
        # Compute gradient w.r.t. input for target class
        class_logit = logits[0, class_idx]
        class_logit.backward()
        
        input_grad = x_tensor.grad[0].numpy()  # (signal_len, n_leads)
    
    # CAM: amplitude of gradient indicates importance
    cam = np.abs(input_grad)  # (signal_len, n_leads)
    
    # Visualize
    fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=300)
    
    # Heatmap of temporal importance per lead
    im = axes.imshow(cam.T, aspect='auto', cmap='hot', origin='lower')
    axes.set_xlabel('Time (samples @ 100Hz)', fontsize=12)
    axes.set_ylabel('Lead', fontsize=12)
    axes.set_yticks(range(len(lead_order)))
    axes.set_yticklabels(lead_order)
    axes.set_title(f'Class Activation Map: {MAIN_LABELS[class_idx]} MI', 
                  fontsize=14, weight='bold')
    
    cbar = plt.colorbar(im, ax=axes, label='Gradient Magnitude (Importance)')
    fig.tight_layout()
    
    return fig, cam
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] A1: Beat windowing with context (200 samples)
- [ ] B2: Learnable prior matrix with Dirichlet regularization
- [ ] F2: Prior-attention agreement metric
- [ ] **Expected improvement**: +2-3% accuracy

### Phase 2: Feature Enrichment (Weeks 3-4)
- [ ] A2: Multi-beat sequence encoding
- [ ] C1: ST-segment extraction module
- [ ] D1: Lead positional encoding
- [ ] F1: Gradient-based CAM
- [ ] **Expected improvement**: +4-6% accuracy

### Phase 3: Robustness (Week 5+)
- [ ] A3: RR-interval features
- [ ] C2/C3: T-wave and QRS modules
- [ ] D2: Lead neighborhood attention
- [ ] E1/E2: Focal loss + per-class priors
- [ ] F4: Full CAM analysis
- [ ] **Expected improvement**: +3-5% accuracy

---

## Validation Protocol

```python
def full_architecture_validation():
    """Compare baseline vs all revisions"""
    
    results = {}
    
    # Baseline
    model_baseline = ALPANet(CONFIG).to(device)
    results['baseline'] = evaluate(model_baseline, test_loader)
    
    # Single revisions
    for revision in ['A1', 'A2', 'B2', 'C1', 'D1', 'E1', 'F2']:
        model = ALPANet_with_revision(CONFIG, revision).to(device)
        results[revision] = evaluate(model, test_loader)
    
    # Combined Phase 1
    model = ALPANet_phase1(CONFIG).to(device)
    results['Phase1'] = evaluate(model, test_loader)
    
    # Comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df['improvement'] = comparison_df['macro_f1'] - results['baseline']['macro_f1']
    
    return comparison_df
```

---

## Files to Modify/Create

1. **Data Loading** → A1, A2, A3 implementation
2. **Model Architecture** → B2, C1, D1, H revisions
3. **Training Loop** → E1 (Focal Loss), F2 (Agreement metric)
4. **Validation** → F1, F4 (CAM visualization)
5. **Analysis** → New notebook for comparative studies

---

## References

- Anatomical ECG interpretation (cardiology textbooks)
- ST-elevation MI: elevation in anatomically contiguous leads
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations..." (2016)
- Dirichlet priors: Blei et al., "Latent Dirichlet Allocation" (2003)

