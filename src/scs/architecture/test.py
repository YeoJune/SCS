import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from transformers import T5Config

# io.py와 transformer.py가 이 스크립트와 같은 디렉토리에 있다고 가정합니다.
# 경로가 다른 경우 sys.path.append() 등을 사용하여 경로를 추가해야 합니다.
try:
    from .transformer import (
        TransformerEncoder, TransformerEncoderLayer,
        TransformerDecoder, TransformerDecoderLayer
    )
    from .io import InputInterface, OutputInterface
except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure 'io.py' and 'transformer.py' are in the same directory as this script.")
    exit()

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
PAD_TOKEN_ID = 0

# --- t5-small 모델에서 파라미터 자동 로드 ---
print(f"Loading configuration from '{T5_MODEL_NAME}'...")
try:
    config = T5Config.from_pretrained(T5_MODEL_NAME)
    EMBEDDING_DIM = config.d_model
    VOCAB_SIZE = config.vocab_size
except Exception as e:
    print(f"Could not load T5 config. Using default values. Error: {e}")
    EMBEDDING_DIM = 512
    VOCAB_SIZE = 32128

# --- 시뮬레이션 파라미터 ---
GRID_SIZE = 32
BATCH_SIZE = 128
NUM_BATCHES = 200 # 샘플 수를 늘림 (총 128 * 200 = 25,600개 샘플)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "simulation_results_t5_small_safe"
VIS_SAMPLES = 5000 # 시각화에 사용할 샘플 수

# 시뮬레이션 결과 저장 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSimulation Environment:")
print(f"  - Device: {DEVICE}")
print(f"  - Total Batches: {NUM_BATCHES} (Batch Size: {BATCH_SIZE})")
print(f"  - Total Vector Samples: {NUM_BATCHES * BATCH_SIZE * WINDOW_SIZE * 2 + NUM_BATCHES * BATCH_SIZE}")
print(f"  - Embedding Dim: {EMBEDDING_DIM}")
print(f"Results will be saved in '{OUTPUT_DIR}' directory.")


# ============================================================================
# 모델 초기화 및 가중치 이식 (MODEL SETUP & TRANSPLANT)
# ============================================================================
print("\nInitializing models and attempting to transplant T5 weights...")

# InputInterface와 OutputInterface를 초기화합니다.
# t5_model_name 인자를 전달하면 내부적으로 가중치 이식이 시도됩니다.
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
    t5_model_name=T5_MODEL_NAME,
    device=DEVICE
).to(DEVICE).eval()

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
    t5_model_name=T5_MODEL_NAME,
    transplant_cross_attention=TRANSPLANT_CROSS_ATTENTION,
    device=DEVICE
).to(DEVICE).eval()


# ============================================================================
# 스트리밍 방식 시뮬레이션 (MEMORY-EFFICIENT SIMULATION)
# ============================================================================

# 1. 통계량 누적을 위한 변수 초기화
stats = {
    name: {'sum': 0.0, 'sum_sq': 0.0, 'norm_sum': 0.0, 'norm_sum_sq': 0.0, 'count': 0}
    for name in ["encoder_output", "hidden_vector", "decoder_output"]
}

# 2. 시각화를 위한 데이터 샘플링 변수 초기화 (저수지 샘플링)
samples = {
    name: torch.empty(VIS_SAMPLES, EMBEDDING_DIM)
    for name in ["encoder_output", "hidden_vector", "decoder_output"]
}
total_samples_seen = {name: 0 for name in samples}

print("\nStarting memory-efficient simulation...")
for i in tqdm(range(NUM_BATCHES), desc="Streaming Batches"):
    with torch.no_grad():
        # --- 데이터 생성 ---
        token_window = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, WINDOW_SIZE), device=DEVICE)
        spike_prob = torch.rand(BATCH_SIZE, 1, 1, device=DEVICE) * 0.15 + 0.05
        grid_spikes = (torch.rand(BATCH_SIZE, GRID_SIZE, GRID_SIZE, device=DEVICE) < spike_prob).float()
        decoder_input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, WINDOW_SIZE), device=DEVICE)

        # --- 모델 실행 ---
        # InputInterface의 encoder_output 추출
        token_embeds = input_interface.token_embedding(token_window)
        encoder_output_vecs = input_interface.transformer_encoder(token_embeds)

        # OutputInterface의 hidden_vector 및 decoder_output 추출
        hidden_vector = output_interface._create_hidden_vector(grid_spikes)
        output_interface.update_hidden_window(grid_spikes)
        decoder_output_vecs = output_interface(decoder_input_ids)

        # 처리할 벡터들 딕셔너리
        vectors_to_process = {
            "encoder_output": encoder_output_vecs.reshape(-1, EMBEDDING_DIM),
            "hidden_vector": hidden_vector,
            "decoder_output": decoder_output_vecs.reshape(-1, EMBEDDING_DIM)
        }

        # --- 각 벡터에 대해 통계량 및 샘플 업데이트 ---
        for name, data in vectors_to_process.items():
            data_cpu = data.cpu()

            # 온라인 통계량 업데이트
            stats[name]['count'] += len(data_cpu)
            stats[name]['sum'] += data_cpu.sum().item()
            stats[name]['sum_sq'] += torch.sum(data_cpu**2).item()
            norms = torch.linalg.norm(data_cpu.float(), dim=1) # linalg.norm은 float을 선호
            stats[name]['norm_sum'] += norms.sum().item()
            stats[name]['norm_sum_sq'] += torch.sum(norms**2).item()

            # 저수지 샘플링 (Reservoir Sampling)
            for item in data_cpu:
                total_samples_seen[name] += 1
                if total_samples_seen[name] <= VIS_SAMPLES:
                    samples[name][total_samples_seen[name] - 1] = item
                else:
                    j = np.random.randint(0, total_samples_seen[name])
                    if j < VIS_SAMPLES:
                        samples[name][j] = item

print("Simulation finished.")

# ============================================================================
# 최종 통계량 계산 및 저장 (FINAL STATISTICS)
# ============================================================================
stats_data = []
for name, s in stats.items():
    if s['count'] == 0: continue
    
    # Per-element stats
    n_elements = s['count'] * EMBEDDING_DIM
    mean = s['sum'] / n_elements
    # 온라인 분산 계산 (더 안정적)
    std = np.sqrt(max(0, s['sum_sq'] / n_elements - mean**2))

    # Per-vector stats
    n_vectors = s['count']
    norm_mean = s['norm_sum'] / n_vectors
    norm_std = np.sqrt(max(0, s['norm_sum_sq'] / n_vectors - norm_mean**2))

    stats_data.append({
        "Vector Type": name,
        "Total Samples": f"{s['count']}",
        "Mean (element-wise)": f"{mean:.4f}",
        "Std Dev (element-wise)": f"{std:.4f}",
        "Mean Norm (vector-wise)": f"{norm_mean:.4f}",
        "Std Dev Norm (vector-wise)": f"{norm_std:.4f}"
    })

stats_df = pd.DataFrame(stats_data)
stats_file = os.path.join(OUTPUT_DIR, "statistics_summary.csv")
stats_df.to_csv(stats_file, index=False)
print(f"\nStatistics saved to '{stats_file}':")
print(stats_df.to_string())

# ============================================================================
# 분포 시각화 (VISUALIZATION - NO SKLEARN)
# ============================================================================
print(f"\nGenerating visualizations from up to {VIS_SAMPLES} samples...")

# 공통 시각화 함수
def create_plots(name, data_tensor):
    num_actual_samples = min(total_samples_seen[name], VIS_SAMPLES)
    if num_actual_samples == 0:
        print(f"  - No samples for '{name}', skipping plots.")
        return
        
    data_np = data_tensor[:num_actual_samples].numpy()
    
    # 1. Element Value Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data_np.flatten(), bins=100, kde=True, color='skyblue')
    plt.title(f"Distribution of Element Values\n({name})", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_elements.png"), bbox_inches='tight')
    plt.close()

    # 2. Vector L2 Norm Distribution
    plt.figure(figsize=(8, 6))
    norms = np.linalg.norm(data_np, axis=1)
    sns.histplot(norms, bins=100, kde=True, color='salmon')
    plt.title(f"Distribution of Vector L2 Norms\n({name})", fontsize=16)
    plt.xlabel("L2 Norm", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_norms.png"), bbox_inches='tight')
    plt.close()
    
    # 3. 2D PCA Visualization (using torch.pca_lowrank)
    try:
        sample_tensor = data_tensor[:num_actual_samples]
        centered_data = sample_tensor - sample_tensor.mean(dim=0, keepdim=True)
        U, S, V = torch.pca_lowrank(centered_data.to(DEVICE), q=2) # PCA는 GPU에서 더 빠를 수 있음
        data_pca = torch.matmul(centered_data.to(DEVICE), V).cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.3, s=5, color='cornflowerblue')
        plt.title(f"2D PCA Visualization (using torch.pca_lowrank)\n({name})", fontsize=16)
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.savefig(os.path.join(OUTPUT_DIR, f"pca_2d_{name}.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  - Could not generate PCA plot for '{name}': {e}")


# 각 벡터 타입에 대해 시각화 실행
for name, data_tensor in tqdm(samples.items(), desc="Creating Plots"):
    create_plots(name, data_tensor)

print("\nAll tasks finished successfully.")