This project focuses on environmental sound classification (ESC) using deep learning techniques. We build a complete pipeline to classify real-world sounds from the ESC-50 dataset by transforming raw audio signals into Log-Mel Spectrograms and learning discriminative patterns using a Convolutional Neural Network (CNN).

Key Features

- Audio preprocessing: waveform → STFT → Log-Mel Spectrogram

- Feature engineering: Log-Mel + Delta + Delta-Delta (3-channel input)

- Deep learning model: CNN for time–frequency representation learning

- Data augmentation: SpecAugment (time masking, frequency masking)

- Model explainability: Grad-CAM visualization for interpretability

- Baseline comparison: MFCC + kNN vs CNN model

Goal : The project aims to demonstrate how time–frequency representations combined with CNNs can effectively learn discriminative audio features, improving performance over traditional handcrafted features, and providing interpretable predictions for real-world sound classification systems.

Applications: smart surveillance, smart cities, IoT systems, acoustic monitoring, AI-based audio analytics.
