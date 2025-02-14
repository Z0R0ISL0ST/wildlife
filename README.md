**requirements.txt**
```
tensorflow
numpy
opencv-python
pandas
streamlit
Pillow
scikit-learn
matplotlib
seaborn
```  
**README.md**
```
# WildTrack AI: Wildlife Footprint Classification

## Overview
WildTrack AI is a machine learning-based application that identifies animal species based on footprint images. This project uses a Convolutional Neural Network (CNN) trained on footprint images of lions, coyotes, and elephants.

## Features
- Upload footprint images to predict the species.
- Uses a deep learning model (CNN) for classification.
- Implements data preprocessing and augmentation.
- Deployable as a Streamlit web app.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/wildtrack-ai.git
   cd wildtrack-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset consists of footprint images categorized into:
- Lion
- Coyote
- Elephant

### Dataset Structure
```
footprints_dataset/
│── train/
│   │── lion/
│   │── coyote/
│   │── elephant/
│── validation/
│   │── lion/
│   │── coyote/
│   │── elephant/
```

## Model Training
Train the CNN model using:
```python
python train_model.py
```

## Deployment
The trained model (`wildlife_model.h5`) is used in `app.py` for predictions. The Streamlit app allows users to upload an image and receive a classification result.

## Next Steps
- Enhance the dataset with real-world images.
- Use MobileNetV2 for better accuracy (transfer learning).
- Implement GPS-based wildlife tracking.

## Contributors
- Suprith K

## License
This project is licensed under the MIT License.
```

