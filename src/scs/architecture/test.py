import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from transformers import T5Config, AutoTokenizer

# io.py와 transformer.py가 이 스크립트와 같은 디렉토리에 있다고 가정합니다.
try:
    from transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
    from io import InputInterface, OutputInterface
except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure 'io.py' and 'transformer.py' are in the same directory as this script.")
    exit()

# ============================================================================
# 시뮬레이션 설정
# ============================================================================
T5_MODEL_NAME = "t5-small"
WINDOW_SIZE = 16 # 시퀀스 길이
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
GRID_SIZE = 32
GRID_SPIKE_THRESHOLD = 1.0 # 그리드 스파이크 발화 임계값

print(f"Loading configuration and tokenizer from '{T5_MODEL_NAME}'...")
try:
    config = T5Config.from_pretrained(T5_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    EMBEDDING_DIM = config.d_model
    VOCAB_SIZE = config.vocab_size
    PAD_TOKEN_ID = config.pad_token_id
except Exception as e:
    print(f"Could not load T5 config/tokenizer. Exiting. Error: {e}")
    exit()

BATCH_SIZE = 128
NUM_BATCHES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "simulation_results_real_data_final"
VIS_SAMPLES = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSimulation Environment:")
print(f"  - Device: {DEVICE}, Total Batches: {NUM_BATCHES}, Batch Size: {BATCH_SIZE}")
print(f"Results will be saved in '{OUTPUT_DIR}' directory.")


# ============================================================================
# 샘플 데이터 준비 (실제 텍스트)
# ============================================================================
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "In the beginning God created the heaven and the earth.",
    "Machine learning is a subset of artificial intelligence.",
    "The transformer architecture has revolutionized natural language processing.",
    "What is the capital of France? The capital is Paris.",
    "Photosynthesis is the process used by plants, algae and certain bacteria to convert light energy into chemical energy.",
    "Gravity is the fundamental force of attraction that all matter exerts on all other matter.",
    "The theory of relativity, developed by Albert Einstein, is one of the two pillars of modern physics.",
    "Neuro-symbolic AI combines neural networks with symbolic reasoning.",
    "The human brain is a complex network of billions of neurons and synapses.",
]

# ============================================================================
# 모델 초기화
# ============================================================================
print("\nInitializing models and transplanting T5 weights...")
input_interface = InputInterface(
    vocab_size=VOCAB_SIZE, grid_height=GRID_SIZE, grid_width=GRID_SIZE,
    embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE, encoder_layers=ENCODER_LAYERS,
    encoder_heads=ENCODER_HEADS, dim_feedforward=DIM_FEEDFORWARD, encoder_dropout=ENCODER_DROPOUT,
    input_power=INPUT_POWER, softmax_temperature=SOFTMAX_TEMPERATURE, t5_model_name=T5_MODEL_NAME,
    device=DEVICE
).to(DEVICE).eval()

output_interface = OutputInterface(
    vocab_size=VOCAB_SIZE, grid_height=GRID_SIZE, grid_width=GRID_SIZE,
    pad_token_id=PAD_TOKEN_ID, embedding_dim=EMBEDDING_DIM, window_size=WINDOW_SIZE,
    decoder_layers=DECODER_LAYERS, decoder_heads=DECODER_HEADS,
    dim_feedforward=DIM_FEEDFORWARD, dropout=DECODER_DROPOUT,
    t5_model_name=T5_MODEL_NAME, transplant_cross_attention=TRANSPLANT_CROSS_ATTENTION,
    device=DEVICE
).to(DEVICE).eval()


# ============================================================================
# 스트리밍 방식 시뮬레이션
# ============================================================================
VECTOR_NAMES = [
    "encoder_output_pre_norm", "encoder_output_post_norm",
    "hidden_vector_pre_norm", "hidden_vector_post_norm",
    "decoder_output_pre_norm", "decoder_output_post_norm"
]

stats = {name: {'sum': 0., 'sum_sq': 0., 'norm_sum': 0., 'norm_sum_sq': 0., 'count': 0} for name in VECTOR_NAMES}
samples = {name: torch.empty(VIS_SAMPLES, EMBEDDING_DIM) for name in VECTOR_NAMES}
total_samples_seen = {name: 0 for name in samples}

print("\nStarting simulation with real tokenized data...")
for i in tqdm(range(NUM_BATCHES), desc="Streaming Batches"):
    with torch.no_grad():
        # --- 1. 실제 데이터 기반 입력 생성 ---
        batch_texts = np.random.choice(sample_texts, BATCH_SIZE).tolist()
        inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=WINDOW_SIZE)
        token_window = inputs.input_ids.to(DEVICE)
        
        decoder_input_ids = torch.roll(token_window, shifts=-1, dims=1)
        decoder_input_ids[:, -1] = PAD_TOKEN_ID
        
        # --- 2. 모델 실행 및 벡터 추출 ---
        # Encoder
        token_embeds = input_interface.token_embedding(token_window)
        encoder_output = token_embeds
        for layer in input_interface.transformer_encoder.layers:
            encoder_output = layer(encoder_output)
        encoder_output_pre_norm = encoder_output
        encoder_output_post_norm = input_interface.transformer_encoder.norm(encoder_output_pre_norm)

        # 3. 의미 있는 grid_spikes 생성
        context_vector = encoder_output_post_norm[:, -1, :]
        scaled_pattern = input_interface.forward(token_window) # forward로 직접 생성
        grid_spikes = (scaled_pattern > GRID_SPIKE_THRESHOLD).float()

        # Hidden Vector
        spikes_flat = grid_spikes.view(grid_spikes.shape[0], -1)
        hidden_vector_pre_norm = output_interface.spatial_compressor(spikes_flat)
        hidden_vector_post_norm = output_interface.hidden_norm(hidden_vector_pre_norm)
        output_interface.update_hidden_window(grid_spikes)

        # Decoder
        decoder_output_pre_norm = output_interface(decoder_input_ids, return_pre_norm=True)
        decoder_output_post_norm = output_interface.transformer_decoder.norm(decoder_output_pre_norm)
        
        # 처리할 벡터들 딕셔너리
        vectors_to_process = {
            "encoder_output_pre_norm": encoder_output_pre_norm.reshape(-1, EMBEDDING_DIM),
            "encoder_output_post_norm": encoder_output_post_norm.reshape(-1, EMBEDDING_DIM),
            "hidden_vector_pre_norm": hidden_vector_pre_norm,
            "hidden_vector_post_norm": hidden_vector_post_norm,
            "decoder_output_pre_norm": decoder_output_pre_norm.reshape(-1, EMBEDDING_DIM),
            "decoder_output_post_norm": decoder_output_post_norm.reshape(-1, EMBEDDING_DIM)
        }
        
        # --- 4. 통계량 및 샘플 업데이트 ---
        for name, data in vectors_to_process.items():
            data_cpu = data.cpu()
            stats[name]['count'] += len(data_cpu)
            stats[name]['sum'] += data_cpu.sum().item()
            stats[name]['sum_sq'] += torch.sum(data_cpu**2).item()
            norms = torch.linalg.norm(data_cpu.float(), dim=1)
            stats[name]['norm_sum'] += norms.sum().item()
            stats[name]['norm_sum_sq'] += torch.sum(norms**2).item()
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
# 최종 통계량 계산 및 저장 
# ============================================================================
stats_data = []
for name in VECTOR_NAMES:
    s = stats[name]
    if s['count'] == 0: continue
    n_elements = s['count'] * EMBEDDING_DIM
    mean = s['sum'] / n_elements
    std = np.sqrt(max(0, s['sum_sq'] / n_elements - mean**2))
    n_vectors = s['count']
    norm_mean = s['norm_sum'] / n_vectors
    norm_std = np.sqrt(max(0, s['norm_sum_sq'] / n_vectors - norm_mean**2))
    stats_data.append({
        "Vector Type": name, "Total Samples": f"{s['count']}", "Mean (element-wise)": f"{mean:.4f}",
        "Std Dev (element-wise)": f"{std:.4f}", "Mean Norm (vector-wise)": f"{norm_mean:.4f}",
        "Std Dev Norm (vector-wise)": f"{norm_std:.4f}"
    })
stats_df = pd.DataFrame(stats_data)
stats_file = os.path.join(OUTPUT_DIR, "statistics_summary_real_data.csv")
stats_df.to_csv(stats_file, index=False)
print(f"\nStatistics saved to '{stats_file}':\n{stats_df.to_string()}")

# ============================================================================
# 분포 시각화 (NO SKLEARN)
# ============================================================================
print(f"\nGenerating visualizations from up to {VIS_SAMPLES} samples...")

def create_plots(name, data_tensor):
    num_actual_samples = min(total_samples_seen[name], VIS_SAMPLES)
    if num_actual_samples == 0:
        print(f"  - No samples for '{name}', skipping plots.")
        return
        
    data_np = data_tensor[:num_actual_samples].numpy()
    
    # Element Value Distribution
    plt.figure(figsize=(8, 6)); sns.histplot(data_np.flatten(), bins=100, kde=True, color='skyblue')
    plt.title(f"Distribution of Element Values\n({name})", fontsize=16); plt.xlabel("Value", fontsize=12); plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_elements.png"), bbox_inches='tight'); plt.close()

    # Vector L2 Norm Distribution
    plt.figure(figsize=(8, 6)); norms = np.linalg.norm(data_np, axis=1)
    sns.histplot(norms, bins=100, kde=True, color='salmon'); plt.title(f"Distribution of Vector L2 Norms\n({name})", fontsize=16)
    plt.xlabel("L2 Norm", fontsize=12); plt.ylabel("Density", fontsize=12); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{name}_norms.png"), bbox_inches='tight'); plt.close()
    
    # 2D PCA Visualization
    try:
        sample_tensor = data_tensor[:num_actual_samples]
        centered_data = sample_tensor - sample_tensor.mean(dim=0, keepdim=True)
        U, S, V = torch.pca_lowrank(centered_data.to(DEVICE), q=2)
        data_pca = torch.matmul(centered_data.to(DEVICE), V).cpu().numpy()

        plt.figure(figsize=(8, 8)); plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.3, s=5, color='cornflowerblue')
        plt.title(f"2D PCA Visualization\n({name})", fontsize=16); plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12); plt.grid(True, linestyle='--', alpha=0.6); plt.axis('equal')
        plt.savefig(os.path.join(OUTPUT_DIR, f"pca_2d_{name}.png"), bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"  - Could not generate PCA plot for '{name}': {e}")

for name in tqdm(VECTOR_NAMES, desc="Creating Plots"):
    create_plots(name, samples[name])

print("\nAll tasks finished successfully.")