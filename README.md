# 🔊 Sound Classification Using UrbanSound8K

This project implements a **neural network-based audio classification system** trained on the **UrbanSound8K dataset** to classify **10 types of urban sounds** from environmental audio recordings.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

Urban environments contain diverse sound sources such as sirens, dog barks, and drilling sounds. This project builds a **sound classification model** capable of recognizing such sounds, which has applications in **smart cities, surveillance, environmental monitoring, and assistive technology**.



## 📚 Dataset

* **Dataset:** [UrbanSound8K]()

* **Description:** Contains **8732 labelled sound excerpts (<=4s)** of urban sounds divided into 10 classes:

  * Air Conditioner
  * Car Horn
  * Children Playing
  * Dog Bark
  * Drilling
  * Engine Idling
  * Gun Shot
  * Jackhammer
  * Siren
  * Street Music

* **Format:** WAV audio files categorized in 10 folders with metadata CSV.



## ✨ Features

✅ Load and preprocess audio data using librosa

✅ Extract **Mel-frequency cepstral coefficients (MFCCs)** as features

✅ Train neural network models (CNN or MLP) for classification

✅ Evaluate model performance with accuracy and confusion matrix

✅ Predict the class of new audio samples



## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* `librosa` (audio feature extraction)
* `numpy`
* `pandas`
* `matplotlib`, `seaborn` (visualization)
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Sound-Classification-Using-UrbanSound8K.git
cd Sound-Classification-Using-UrbanSound8K
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download UrbanSound8K dataset**

* Download from [UrbanSound8K official page](https://)
* Extract into a `data/UrbanSound8K/` directory in the project folder.

5. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `Sound_Classification_UrbanSound8K.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Load dataset metadata and audio files
   * Extract MFCC features from audio
   * Build, train, and evaluate the model
   * Make predictions on sample audio files



## 🏗️ Model Architecture

**CNN-based model:**

* Input: MFCC features with shape `(time_steps, n_mfcc, 1)`
* Convolutional layers with ReLU activation
* MaxPooling layers
* Dropout for regularization
* Dense layer with Softmax activation for classification into 10 classes



## 📁 Project Structure

```
Sound-Classification-Using-UrbanSound8K/
 ┣ data/
 ┃ ┗ UrbanSound8K/
 ┣ images/
 ┃ ┗ (plots and sample confusion matrix)
 ┣ Sound_Classification_UrbanSound8K.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

* **Validation Accuracy:** *e.g. 80% (sample value)*
* **Confusion Matrix:** Shows performance across all 10 classes


The model achieves good performance, effectively classifying urban sounds with high accuracy and generalization.



## 🎧 Example Prediction

```python
# Load and preprocess new audio sample
audio, sample_rate = librosa.load('sample.wav', res_type='kaiser_fast')
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled = np.mean(mfccs.T, axis=0)
mfccs_scaled = mfccs_scaled.reshape(1, -1)

# Predict
prediction = model.predict(mfccs_scaled)
predicted_class = np.argmax(prediction, axis=1)
print("Predicted class:", class_labels[predicted_class[0]])
```



## 🤝 Contributing

Contributions are welcome to:

* Implement **data augmentation** for improved model generalization
* Experiment with **Recurrent Neural Networks or CRNNs**
* Deploy as an API or integrate into IoT-based environmental monitoring systems

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐️ If you find this project useful, please give it a star!


