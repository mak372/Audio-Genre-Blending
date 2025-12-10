
# Audio-Genre-Blending

CSE 244A Project

## Project Description
This project implements audio style transfer by directly optimizing Mel-spectrogram values using TensorFlow's automatic differentiation and backpropagation. Unlike traditional neural style transfer approaches that require training generative models, our method:

Treats the Mel-spectrogram itself as a learnable representation
Combines content loss (structural preservation) and style loss (acoustic character transfer)
Uses gradient descent to iteratively refine the spectrogram
Converts the optimized spectrogram back to audio using inverse Mel-transformation and Griffin-Lim phase reconstruction


## Usage

### Running the Notebook

#### Step 1: Open the Jupyter Notebook

**Option A: Local Machine**
```bash
jupyter notebook Audio_Style_Transfer.ipynb
```

**Option B: Google Colab (Recommended)**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open notebook** → **GitHub**
3. Paste your repository URL
4. Select `Audio_Style_Transfer.ipynb`

#### Step 2: Prepare Audio Files

Choose one of the following options:

**Option A: Download from YouTube**
- Use the YouTube downloader cell in the notebook
- Requires full YouTube video URLs **(NOT Shorts)**
- Recommended videos:
  - Content: `https://www.youtube.com/watch?v=kJQP7kiw9Fk` (Despacito)
  - Style: `https://www.youtube.com/watch?v=5qap5aO4i9A` (Lean On)
- The notebook will automatically convert to WAV at 16 kHz sample rate

**Option B: Upload Local Audio Files**
- Supported formats: MP3, WAV, M4A, FLAC
- Upload files to the notebook working directory
- The notebook includes a utility to convert to WAV at 16 kHz sample rate
- Minimum recommended length: 15-30 seconds per audio file

#### Step 3: Run Notebook Cells in Order

Execute cells sequentially as follows:

| Cell Range | Description | Output |
|---|---|---|
| **Cell 1-3** | Import libraries and utility functions | Dependencies loaded, functions defined |
| **Cell 4-6** | Load audio files and convert to Mel-spectrograms | `content_song.wav`, `style_song.wav`, mel-spectrogram visualization |
| **Cell 7-9** | Define loss functions (content + style) | Loss function implementations ready |
| **Cell 10-13** | Training loop with TensorFlow GradientTape | 300 epochs of optimization, loss decreases over time |
| **Cell 14-15** | Convert optimized spectrogram back to audio | Generated WAV file with style transfer applied |
| **Cell 16-17** | Visualization and audio playback | Spectrogram comparison, loss curves, audio player |

#### Step 4: Listen to Results

The notebook generates three audio outputs for comparison:

1. **Baseline Audio**: Original content spectrogram converted to audio (no optimization)
   - Demonstrates Griffin-Lim reconstruction
   - Reference point for evaluation

2. **Optimized Audio**: Spectrogram after 300 epochs of optimization
   - Contains style characteristics
   - Preserves content structure

3. **Visual Comparisons**:
   - Content vs. Style vs. Generated Mel-spectrograms side-by-side
   - Loss curves showing convergence over 300 epochs
   - Spectrogram difference heatmaps

### Example Execution

```python
# Cell 1: Import libraries
import librosa
import numpy as np
import tensorflow as tf
from IPython.display import Audio

# Cell 4: Load audio files
content_audio = load_audio_file("content_song.wav")
style_audio = load_audio_file("style_song.wav")

# Cell 5: Convert to Mel-spectrograms
content_mel = get_mel_spectrogram(content_audio)      # Shape: (128, T)
style_mel = get_mel_spectrogram(style_audio)          # Shape: (128, T)

# Cell 6: Align dimensions
min_time = min(content_mel.shape[1], style_mel.shape[1])
content_mel = content_mel[:, :min_time]
style_mel = style_mel[:, :min_time]

# Cell 10-13: Optimization loop (automatic)
# for epoch in range(300):
#     with tf.GradientTape() as tape:
#         loss = total_loss(generated_mel, content_mel, style_mel)
#     grad = tape.gradient(loss, generated_mel)
#     optimizer.apply_gradients([(grad, generated_mel)])

# Cell 15: Convert to audio
output_audio = mel_to_audio(optimized_mel)            # Returns waveform

# Cell 16-17: Display results
print(f"Content Loss: {content_loss_list[-1]:.4f}")
print(f"Style Loss: {style_loss_list[-1]:.4f}")
# Play: baseline_audio, output_audio, visualizations
```

### Tips for Best Results

1. **Audio Selection**:
   - Choose songs with different genres for noticeable style transfer
   - Keep audio lengths similar (30 sec - 2 min works well)
   - Use clear, high-quality recordings

2. **Hyperparameter Tuning**:
   - Increase style weight (β=0.99) for more style influence
   - Increase content weight (α=0.2) for more structure preservation
   - Extend epochs (500-1000) for finer optimization

3. **Troubleshooting**:
   - If audio quality is poor, check input audio quality
   - If optimization is slow, reduce audio length or FFT size
   - If error occurs, ensure all dependencies are installed

### Output Files

After running the notebook, you'll have:

```
project_directory/
├── content_song.wav              # Downloaded/uploaded content audio
├── style_song.wav                # Downloaded/uploaded style audio
└── generated_audio.wav           # Output audio with style transfer
```

(Plus generated plots and loss curves if saved)
## Contributors
- Makarand Bhalerao
- Anannya P Neog
- Aashritha Sankineni

## Course
CSE 244 – Foundations of Deep Learning
