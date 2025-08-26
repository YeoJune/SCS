import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from transformers import T5Config, T5ForConditionalGeneration

# io.py에서 실제 클래스들을 가져옵니다.
# transformer.py가 같은 디렉토리에 있어야 합니다.
from transformer import (
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
    transplant_t5_encoder_weights, transplant_t5_decoder_weights
)
from io import InputInterface, OutputInterface

# ============================================================================
# 시뮬레이션 설정 (CONFIGURATION FROM YAML)
# ============================================================================
# --- 제공된 설정값 ---
T5_MODEL_NAME = "t5-small"
WINDOW_SIZE = 16
ENCODER_LAYERS = 1
DECODER_LAYERS = 1
ENCODER_HEADS = 8
DECODER_HEADS = 8
ENCODER_DROPOUT = 0.1
DECODER_DROPOUT = 0.1
DIM_FEEDFORWARD = 2048
INPUT_POWER = 0.05
SOFTMAX_TEMPERATURE = 0.1
TRANSPLANT_CROSS_ATTENTION = True
PAD_TOKEN_ID = 0 # T5-small의 pad_token_id

# --- t5-small 모델에서 파라미터 자동 로드 ---
print(f"Loading configuration from '{T5_MODEL_NAME}'...")
try:
    config = T5Config.from_pretrained(T5_MODEL_NAME)
    # YAML 설정이 t5-small과 다를 경우를 대비해, 명시된 값을 우선 사용
    EMBEDDING_DIM = config.d_model if config.d_model == 512 else 512
    VOCAB_SIZE = config.vocab_size
    # t5-small의 실제 헤드 수는 8개가 아닐 수 있으므로, 설정값을 존중
    if ENCODER_HEADS != config.num_heads:
        print(f"Warning: YAML encoder_heads ({ENCODER_HEADS}) != t5-small config ({config.num_heads}). Using YAML value.")
    if DECODER_HEADS != config.num_heads:
        print(f"Warning: YAML decoder_heads ({DECODER_HEADS}) != t5-small config ({config.num_heads}). Using YAML value.")
    if DIM_FEEDFORWARD != config.d_ff:
         print(f"Warning: YAML dim_feedforward ({DIM_FEEDFORWARD}) != t5-small config ({config.d_ff}). Using YAML value.")
except Exception as e:
    print(f"Could not load T5 config. Using YAML values. Error: {e}")
    EMBEDDING_DIM = 512
    VOCAB_SIZE = 32128 # t5-small default

# --- 시뮬레이션 파라미터 ---
GRID_SIZE = 32 # YAML에 없지만, 이전 코드 기준
BATCH_SIZE = 128
NUM_BATCHES = 200 # 샘플 수를 늘림 (총 128 * 200 = 25,600개 샘플)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "simulation_results_t5_small"
PCA_TSNE_SAMPLES = 5000 # 시각화에 사용할 샘플 수

# 시뮬레이션 결과 저장 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSimulation Environment:")
print(f"  - Device: {DEVICE}")
print(f"  - Total Batches: {NUM_BATCHES} (Batch Size: {BATCH_SIZE})")
print(f"  - Total Samples: {NUM_BATCHES * BATCH_SIZE}")
print(f"  - Embedding Dim: {EMBEDDING_DIM}")
print(f"Results will be saved in '{OUTPUT_DIR}' directory.")

# ============================================================================
# 모델 초기화 및 가중치 이식 (MODEL SETUP & TRANSPLANT)
# ============================================================================
print("\nInitializing models and transplanting T5 weights...")

# 1. InputInterface 초기화
input_interface = InputInterface(
    vocab_size=VOCAB_SIZE,
    grid_height=GRID_SIZE,
    grid_width=GRID_SIZE,
    embedding_dim=EMBEDDING_DIM,
    window_size=WINDOW_SIZE,
    encoder_layers=ENCODER_LAYERS,
    encoder_heads=ENCODER_HEADS,
    dim_feedforward=DIM_FEEDFORWARD,
    encoder_dropout=ENCODER_DROPOUT,
    input_power=INPUT_POWER,
    softmax_temperature=SOFTMAX_TEMPERATURE,
    t5_model_name=T5_MODEL_NAME, # T5 가중치 이식
    device=DEVICE
).to(DEVICE).eval()

# 2. OutputInterface 초기화
output_interface = OutputInterface(
    vocab_size=VOCAB_SIZE,
    grid_height=GRID_SIZE,
    grid_width=GRID_SIZE,
    pad_token_id=PAD_TOKEN_ID,
    embedding_dim=EMBEDDING_DIM,
    window_size=WINDOW_SIZE,
    decoder_layers=DECODER_LAYERS,
    decoder_heads=DECODER_HEADS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DECODER_DROPOUT,
    t5_model_name=T5_MODEL_NAME, # T5 가중치 이식
    transplant_cross_attention=TRANSPLANT_CROSS_ATTENTION,
    device=DEVICE
).to(DEVICE).eval()


# ============================================================================
# 데이터 생성 및 시뮬레이션 실행 (SIMULATION RUN)
# ============================================================================
results = {"encoder_output": [], "hidden_vector": [], "decoder_output": []}
print("\nStarting simulation...")
for _ in tqdm(range(NUM_BATCHES), desc="Simulating Batches"):
    with torch.no_grad():
        # --- InputInterface 시뮬레이션 ---
        token_window = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, WINDOW_SIZE), device=DEVICE)
        encoder_output = input_interface.transformer_encoder(
            input_interface.dropout(input_interface.token_embedding(token_window))
        )
        results["encoder_output"].append(encoder_output.reshape(-1, EMBEDDING_DIM).cpu())

        # --- OutputInterface 시뮬레이션 ---
        spike_prob = torch.rand(BATCH_SIZE, 1, 1, device=DEVICE) * 0.15 + 0.05
        grid_spikes = (torch.rand(BATCH_SIZE, GRID_SIZE, GRID_SIZE, device=DEVICE) < spike_prob).float()
        
        hidden_vector = output_interface._create_hidden_vector(grid_spikes)
        results["hidden_vector"].append(hidden_vector.cpu())

        output_interface.reset_state(BATCH_SIZE)
        output_interface.hidden_window = hidden_vector.unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
        
        decoder_input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, WINDOW_SIZE), device=DEVICE)
        target_embeds = output_interface.token_embedding(decoder_input_ids)
        
        tgt_len = target_embeds.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=DEVICE), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        decoder_output = output_interface.transformer_decoder(
            tgt=target_embeds, memory=output_interface.hidden_window, tgt_mask=causal_mask
        )
        results["decoder_output"].append(decoder_output.reshape(-1, EMBEDDING_DIM).cpu())

# 모든 배치 결과 합치기
for key in results:
    results[key] = torch.cat(results[key], dim=0).numpy()
print("Simulation finished.")

# ============================================================================
# 통계량 계산 및 저장 (STATISTICS)
# ============================================================================
stats_data = []
for name, data in results.items():
    stats_data.append({
        "Vector Type": name,
        "Shape": str(data.shape),
        "Mean (element-wise)": f"{np.mean(data):.4f}",
        "Std Dev (element-wise)": f"{np.std(data):.4f}",
        "Mean Norm (vector-wise)": f"{np.mean(np.linalg.norm(data, axis=1)):.4f}",
        "Std Dev Norm (vector-wise)": f"{np.std(np.linalg.norm(data, axis=1)):.4f}"
    })

stats_df = pd.DataFrame(stats_data)
stats_file = os.path.join(OUTPUT_DIR, "statistics_summary.csv")
stats_df.to_csv(stats_file, index=False)
print(f"\nStatistics saved to '{stats_file}':")
print(stats_df.to_string())

# ============================================================================
# 분포 시각화 (VISUALIZATION)
# ============================================================================
print(f"\nGenerating visualizations (saved in '{OUTPUT_DIR}')...")
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 공통 시각화 함수
def create_plots(name, data):
    # 1. Element Value Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data.flatten(), bins=100, kde=True, color='skyblue')
    plt.title(f"Distribution of Element Values\n({name})", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_elements.png"), bbox_inches='tight')
    plt.close()

    # 2. Vector L2 Norm Distribution
    plt.figure(figsize=(8, 6))
    norms = np.linalg.norm(data, axis=1)
    sns.histplot(norms, bins=100, kde=True, color='salmon')
    plt.title(f"Distribution of Vector L2 Norms\n({name})", fontsize=16)
    plt.xlabel("L2 Norm", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_norms.png"), bbox_inches='tight')
    plt.close()

    # 데이터 샘플링 for PCA/t-SNE
    if len(data) > PCA_TSNE_SAMPLES:
        indices = np.random.choice(len(data), PCA_TSNE_SAMPLES, replace=False)
        sample = data[indices]
    else:
        sample = data
    
    # 3. 2D PCA Visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(sample)
    plt.figure(figsize=(8, 8))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.3, s=5, color='cornflowerblue')
    plt.title(f"2D PCA Visualization\n({name})", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, f"pca_2d_{name}.png"), bbox_inches='tight')
    plt.close()

    # 4. 2D t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    data_tsne = tsne.fit_transform(sample)
    plt.figure(figsize=(8, 8))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], alpha=0.3, s=5, color='mediumseagreen')
    plt.title(f"2D t-SNE Visualization\n({name})", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, f"tsne_2d_{name}.png"), bbox_inches='tight')
    plt.close()

# 각 벡터 타입에 대해 시각화 실행
for name, data in tqdm(results.items(), desc="Creating Plots"):
    create_plots(name, data)

print("All tasks finished successfully.")