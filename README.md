## Driver Drowsiness Detection System 

### Project Overview
This project delivers a robust, real-time solution for mitigating driving risks by detecting driver fatigue. It leverages **Deep Learning (CNN)** for eye state classification, integrated into a scalable **Production Architecture** and deployed via **FastAPI** on Azure.

| Component | Technology | Key Feature |
| :--- | :--- | :--- |
| **Model** | TensorFlow/Keras (CNN) | Binary classification (Awake/Sleepy) |
| **Backend API** | FastAPI, Docker | High-performance, containerized prediction endpoint |
| **Real-Time Processing** | OpenCV, Python | Live frame capture and eye region detection |
| **Deployment** | Azure Container Instances (ACI) | Scalable and reliable cloud hosting |

______________________________

#### 1. Project Structure & Repository Layout


This structure facilitates clear separation of concerns, supporting training, API serving, and deployment.

<pre>
.
│
├── dataset/                     ← Training data
│   ├── awake/                   ← Images of open eyes
│   └── sleepy/                  ← Images of closed eyes
│
├── dl-model/                    ← Model and training scripts
│   ├── eye_state_model.keras    ← Trained CNN model
│   ├── train_model.ipynb        ← Model training notebook
│   └── test_img_open_close.jpg  ← Sample test image
│
├── api/                         ← Backend (model deployment)
│   ├── app.py                   ← FastAPI application (prediction endpoint)
│   ├── Dockerfile               ← Container configuration
│   ├── requirements.txt         ← Backend dependencies
│   ├── start.sh                 ← Startup script (for Azure)
│   └── test_api.ipynb           ← Local API testing notebook
│
├── ui/                          ← Frontend (user interface)
│   └── main.py                  ← Gradio interface (webcam + API requests)
│
├── .gitignore                   ← Git ignored files
├── README.md                    ← Project documentation
└── requirements.txt             ← Global dependencies         
</pre>

#### 2. Data Source
The system is trained on the high-quality **MRL Eye Dataset** from Kaggle, ensuring robust generalization across different eye conditions.
- **Source:** [Kaggle - MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset?resource=download)
- **Data Used:** $\approx 50,000$ infrared images categorized into **Awake** and **Sleepy** classes.

#### 3. Deep Learning Model Training
The core of the system is a **Convolutional Neural Network (CNN)** optimized for fast and accurate binary classification.

* **Pre-processing**: Image normalization, resizing, and cleaning.
* **Architecture**: Custom CNN designed for low latency inference.
* **Evaluation**: Rigorous testing focused on high **Accuracy**, **Precision**, and **Recall** to minimize false negatives (missed drowsiness).

---

#### 4. Production Architecture & Scalable Deployment

##### 4.1. End-to-End Deployment Pipeline

This diagram maps the complete Machine Learning lifecycle, from data ingestion to cloud deployment. The model is containerized using **Docker** and deployed on **Azure Container Instances (ACI)**, exposed via a production-grade **FastAPI** service.
![End-to-End Deep Learning Project Workflow](https://github.com/user-attachments/assets/4ca8b4c4-274f-4db6-9bda-2fc59cc69bf1)

##### 4.2. Real-Time Drowsiness Detection System Flow

This critical flow illustrates the real-time mechanism. **OpenCV** extracts the eye region from live camera frames, which is sent to the model for prediction. A robust **timing mechanism** is implemented: an alert is triggered only if the eyes are classified as closed for a consecutive period of $\ge 3$ seconds, significantly reducing false positives.
![System Workflow](https://github.com/user-attachments/assets/83e04d49-8daf-4f32-8629-93c10c275c2b)

##### 4.3. Interactive User Interface (Gradio)

The application features a user-friendly interface built with **Gradio**, enabling real-time detection via webcam and visual feedback.

<img width="1874" height="750" alt="Gradio Interface Screenshot" src="https://github.com/user-attachments/assets/19a82c1e-e5bf-4b2e-b0a4-6af779b52734" />

---

#### 5. Real-Time Detection Examples

| Case | Description | Image |
| :--- | :--- | :--- |
| **a. Eyes Open** | Normal state - No alert. | ![eyes_open](https://github.com/user-attachments/assets/307e8237-6e9b-4958-b85c-b4df57ffaa28) |
| **b. Partial Closure** | One eye closed - No alert (maintaining driver focus). | ![one_eye_closed](https://github.com/user-attachments/assets/c7d34f20-6dde-4e6e-97cd-58a861581a77) |
| **c. Eyes Closed** | Drowsiness detected - **ALERT triggered** (after $\ge 3$ seconds). | ![eyes_closed](https://github.com/user-attachments/assets/662ee59b-8372-4ab5-9eb4-9bc468d8045a) |
| **d. No Eyes Detected**| Eyes obscured or user looking away - **ALERT triggered** (safety default). | ![no_eyes](https://github.com/user-attachments/assets/48c4f8fd-7a69-440b-ac46-b64806009889) |

---
*Developed by Yassine DARIF | INSEA - 2025*
