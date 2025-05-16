# Chest X-ray Diagnosis using CNN (AlexNet)

This project explores the application of Convolutional Neural Networks (CNNs) to automate the diagnosis of chest X-ray images, specifically classifying them into "Normal" or "Pneumonia". Leveraging the pre-trained AlexNet architecture and the Harvard Chest X-ray Dataset 2, we fine-tuned the model for high accuracy in medical image classification.

## 🩺 Motivation
Manual diagnosis of chest X-rays is time-consuming and prone to human error. Our objective is to assist radiologists by providing a fast and reliable AI-based diagnostic tool.

## 📊 Dataset
- **Harvard Chest X-ray Dataset 2**
- Contains labeled chest X-ray images with pathologies.

## 🧠 Model
- **AlexNet** (Pretrained on ImageNet)
- Final layer modified for binary classification (Normal vs Pneumonia).

## 🛠️ Methodology
1. **Data Preprocessing**: Resizing, normalization, data augmentation (rotation, flipping).
2. **Training**: 
   - Optimizer: SGD / Adam
   - Loss Function: CrossEntropyLoss
   - Epochs: 25
   - Batch Size: 32
   - Learning Rate Scheduler
3. **Fine-tuning** on train/val/test splits.

## 🏆 Results & Observations
- Good generalization on unseen test data.
- Data augmentation improved performance.
- Some confusion in early-stage pneumonia due to visual similarities.

## ✅ Conclusion
- Demonstrated feasibility of AI-assisted radiological diagnosis.
- Transfer learning significantly boosted performance and reduced training time.

## 🚀 Future Scope
- Try deeper models like ResNet, DenseNet.
- Multi-label classification (e.g., TB, COVID-19).
- Real-time deployment via web/mobile interfaces.
- Explainability tools like Grad-CAM.
