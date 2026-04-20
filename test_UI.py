import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import gradio as gr
import librosa
import librosa.display
import numpy as np
from resnet import MainNetwork 

print("Loading Models and System Interface...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update model paths
MODEL_PATHS = {
    "Vietnamese (VIVOS)": os.path.join("weights", "best_audio_model.pth"),
    "English (LJSpeech)": os.path.join("weights", "best_ljspeech_model.pth"),
    "Music": os.path.join("weights", "best_music_model.pth")
}

def load_model(path):
    model = MainNetwork().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    return None

# Load all available models
loaded_models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

# Calculate LSD and SNR
def calculate_lsd(target, predicted):
    n_fft = 2048
    hop_length = 512
    if torch.is_tensor(target): target = target.squeeze().numpy()
    if torch.is_tensor(predicted): predicted = predicted.squeeze().numpy()
    
    S_target = np.abs(librosa.stft(target, n_fft=n_fft, hop_length=hop_length))
    S_pred = np.abs(librosa.stft(predicted, n_fft=n_fft, hop_length=hop_length))

    log_S_target = 20 * np.log10(S_target + 1e-8)
    log_S_pred = 20 * np.log10(S_pred + 1e-8)
    dist = np.sqrt(np.mean((log_S_target - log_S_pred)**2, axis=0))
    return np.mean(dist)

def calculate_snr(target, predicted):
    if torch.is_tensor(target): target = target.squeeze().numpy()
    if torch.is_tensor(predicted): predicted = predicted.squeeze().numpy()
    
    t_norm = target / (np.max(np.abs(target)) + 1e-8)
    p_norm = predicted / (np.max(np.abs(predicted)) + 1e-8)

    noise = t_norm - p_norm
    sig_pwr = np.sum(t_norm ** 2)
    noise_pwr = np.sum(noise ** 2)
    return 10 * np.log10(sig_pwr / (noise_pwr + 1e-8))

# Draw Spectrograms
def plot_spectrogram(wav_tensor, sr, title, lsd_score=None, snr_score=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Librosa Specshow for High-Res plotting
    y_np = wav_tensor.numpy()[0]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_np)), ref=np.max)
    
    # Render with frequency limit based on Sample Rate
    display_limit = 22050 if sr == 44100 else 8000
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_ylim(0, display_limit)
    
    # Dynamic Titles
    if lsd_score is not None and snr_score is not None:
        full_title = f"{title}\n(LSD: {lsd_score:.2f} | SNR: {snr_score:.2f}dB)"
    elif lsd_score is not None:
        full_title = f"{title}\n(LSD: {lsd_score:.2f})"
    else:
        full_title = title
        
    ax.set_title(full_title, fontsize=10, color='white', fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', color='white', fontsize=8)
    ax.set_xlabel('Time (s)', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=8)
    
    # UI Theming
    fig.patch.set_facecolor('#1e1e1e') 
    ax.set_facecolor('#1e1e1e')
    plt.tight_layout()
    return fig

# Audio Processing engine
def process_audio(input_file, model_choice):
    if input_file is None:
        raise gr.Error("Please upload an audio file")
         
    current_model = loaded_models.get(model_choice)
    if current_model is None:
         raise gr.Error(f"Model weights for {model_choice} not found. Please check the weights folder.")

    # Set dynamic sample rates
    if model_choice == "Music":
        high_sr = 44100
        low_sr = 22050
    else:
        high_sr = 16000
        low_sr = 8000

    # Librosa read 
    y_original, sr = librosa.load(input_file, sr=high_sr)
    if len(y_original.shape) > 1: y_original = y_original[0] 
    
    # Downsample via Librosa
    y_lq_low = librosa.resample(y=y_original, orig_sr=high_sr, target_sr=low_sr)
    
    # To Tensor
    wav_truth_high = torch.from_numpy(y_original).unsqueeze(0)
    wav_lq_low = torch.from_numpy(y_lq_low).unsqueeze(0)
    
    # Upsample via Torchaudio
    wav_blurry = T.Resample(low_sr, high_sr)(wav_lq_low)
    
    # Equalize lengths
    min_len = min(wav_truth_high.shape[1], wav_blurry.shape[1])
    wav_truth_high, wav_blurry = wav_truth_high[:, :min_len], wav_blurry[:, :min_len]
    
    # AI enhancement
    with torch.no_grad():
        ai_wav = current_model(wav_blurry.to(device).unsqueeze(0)).squeeze(0).cpu()
    ai_wav = ai_wav[:, :min_len]
    
    # Calculate scores
    lsd_blurry = calculate_lsd(wav_truth_high, wav_blurry)
    lsd_ai = calculate_lsd(wav_truth_high, ai_wav)
    improvement = lsd_blurry - lsd_ai
    
    snr_blurry = calculate_snr(wav_truth_high, wav_blurry)
    snr_ai = calculate_snr(wav_truth_high, ai_wav)
    snr_improvement = snr_ai - snr_blurry 
    
    # Export cache files 
    out_truth = "01_truth.wav"
    out_blurry = "02_blurry.wav"
    out_ai = "03_ai.wav"
    torchaudio.save(out_truth, wav_truth_high, high_sr)
    torchaudio.save(out_blurry, wav_blurry, high_sr)
    torchaudio.save(out_ai, ai_wav, high_sr)
    
    # Render spectrograms
    fig_truth = plot_spectrogram(wav_truth_high, high_sr, "1. GROUND TRUTH", lsd_score=0.0) 
    fig_blurry = plot_spectrogram(wav_blurry, high_sr, "2. BLURRY", lsd_score=lsd_blurry, snr_score=snr_blurry)
    fig_ai = plot_spectrogram(ai_wav, high_sr, "3. AI RESTORATION", lsd_score=lsd_ai, snr_score=snr_ai)
    
    # Dynamic Status Report
    status_msg = (
        f"Doneeeeeeeee\n"
        f"LSD reduced from {lsd_blurry:.2f} to {lsd_ai:.2f} (Improvement: {improvement:.2f})\n"
        f"SNR increased from {snr_blurry:.2f}dB to {snr_ai:.2f}dB (Improvement: +{snr_improvement:.2f}dB)"
    )
    
    return out_truth, out_blurry, out_ai, status_msg, fig_truth, fig_blurry, fig_ai

# UI
custom_theme = gr.themes.Monochrome(
    primary_hue="blue", 
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="HUST Audio AI") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>PROJECT: AUDIO SUPER-RESOLUTION</h1>
            <p><i>By: Dao Tien Dung, Nguyen Thi Hong Nhung - HUST | Using structured Deep ResNet 1D</i></p>
        </div>
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("AI restorer"):
            with gr.Row():
                # Input and Config column
                with gr.Column(scale=1):
                    gr.Markdown("### Inputs and Configuration")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value="Vietnamese (VIVOS)",
                        label="Choose Expert Model",
                        info="The system will automatically route the file to the appropriate model"
                    )
                    input_audio = gr.Audio(label="Upload file / Record voice", type="filepath")
                    run_btn = gr.Button("START RESTORATION", variant="primary", size="lg")
                    status_box = gr.Textbox(label="System Status and Quality Report", interactive=False, lines=4)

                # Results column
                with gr.Column(scale=2):
                    gr.Markdown("### Results Analysis (Ground Truth vs Blurry vs AI)")
                    
                    # Audio players
                    with gr.Row():
                        out_truth_audio = gr.Audio(label="1. Ground Truth", interactive=False)
                        out_blurry_audio = gr.Audio(label="2. Blurry", interactive=False)
                        out_ai_audio = gr.Audio(label="3. AI", interactive=False)
                    
                    # Spectrograms
                    with gr.Row():
                        plot_truth = gr.Plot(label="Original")
                        plot_blurry = gr.Plot(label="Blurry")
                        plot_ai = gr.Plot(label="AI")

        with gr.TabItem("Structure and Data"):
            gr.Markdown(
                """
                ### System Specifications
                - **Neural Network:** 1D Convolutional Neural Network (CNN) with Residual blocks (ResNet)
                - **Loss Function:** L1 Loss (Mean Absolute Error)
                - **Evaluation Index:** - **SNR (Signal-to-Noise Ratio):** Evaluates time-domain energy
                    - **LSD (Log-Spectral Distance):** Evaluates frequency-domain sharpness
                - **Multi-Expert Strategy:** Separates weights for Vietnamese, English, and Music to avoid domain interference.
                """
            )

    run_btn.click(
        fn=process_audio,
        inputs=[input_audio, model_dropdown],
        outputs=[out_truth_audio, out_blurry_audio, out_ai_audio, status_box, plot_truth, plot_blurry, plot_ai]
    )

if __name__ == "__main__":
    demo.launch()