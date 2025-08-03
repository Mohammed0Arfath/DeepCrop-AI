Of course. Based on the detailed information in your hackathon submission document (`AgriThon 2.pdf`), here is a professional and comprehensive README file. It combines the structure of your draft with the specific details, code, and results from your report to create a polished and highly relevant document.

-----

# DeepCrop AI 1.0: Multimodal Crop Disease and Insect Detection

[](https://vit.ac.in/)
[](https://www.python.org/)
[](https://pytorch.org/)
[](https://opensource.org/licenses/MIT)

**DeepCrop AI 1.0** is the official Round 1 submission for **AgriThon 2.0** by **Team DeepCrop 1.0**. The project presents a robust multimodal pipeline for identifying sugarcane diseases and pest infestations by intelligently fusing computer vision with symptom-based textual analysis.

This system was developed for the hackathon organized by the **School of Computer Science and Information Systems, VIT, Vellore**, and sponsored by the **Department of Biotechnology, Govt. of India**.

-----
<img width="2816" height="1536" alt="Gemini_Generated_Image_ip931sip931sip93 (1)" src="https://github.com/user-attachments/assets/aecfb77d-3e94-4bdf-b5cc-6141e3be4c0a" />

## 🏛️ System Architecture & Pipeline

Our solution follows a structured, end-to-end pipeline from data annotation to a final, fused prediction.

1.  **Annotation Phase:** The initial dataset of 50 disease and 50 insect images provided by the organizers was annotated using **CVAT (offline Docker setup)**.
      * **Crop Insects:** Annotated with **Bounding Boxes** for object detection.
      * **Crop Diseases:** Annotated with **Segmentation Masks** for precise localization.
2.  **Augmentation Phase:** The annotated datasets were augmented using techniques like rotation, flipping, color jittering, and contrast adjustments to expand the dataset to 150 images per class and improve model robustness.
3.  **Model Training Phase:** Four specialized models were trained:
      * **YOLOv8s-seg:** Trained on segmentation masks to identify crop diseases.
      * **YOLOv8s:** Trained on bounding boxes to detect crop insects.
      * **TabNet Disease Classifier:** Trained on a synthetic CSV of symptom-based Yes/No questions to predict disease presence.
      * **TabNet Insect Classifier:** Trained similarly on insect-related symptom questions.
4.  **Inference & Fusion Phase:** In the final step, predictions from all four models are aggregated using a strict fusion logic to deliver a final, unified diagnosis.

<img width="3840" height="3176" alt="DeepCrop_Architecture Diagram" src="https://github.com/user-attachments/assets/67ba95cc-d016-4c9a-84b2-6a029114cd8b" />

*(This diagram is a representation of the architecture)*

-----

## ✨ Key Features

  - **🧠 Multimodal Fusion Engine:** The core of our system. A final "Present" verdict is given only if **both** the computer vision model (YOLO) and the questionnaire model (TabNet) return a positive result, minimizing false positives.
  - **🌿 Precise Disease Segmentation:** Uses `YOLOv8s-seg` to not just detect, but accurately outline the exact infected regions on leaves.
  - **🐛 Robust Insect Detection:** Employs `YOLOv8s` to identify and locate insect pests with bounding boxes.
  - **📋 Interactive Symptom-Based Diagnosis:** Leverages two `TabNet` models that guide users through a series of Yes/No questions to diagnose issues based on textual symptoms.
  - **🧪 End-to-End Reproducibility:** The entire workflow, from CVAT setup to running terminal predictions, is fully documented for easy replication.
  - **📊 Comprehensive Performance Metrics:** Includes detailed training graphs, confusion matrices, and performance tables for each model component.

-----

## 🛠️ Tech Stack

| Component                       | Technology / Library                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| **Annotation** | **CVAT (Computer Vision Annotation Tool)**, Docker                                  |
| **Image Augmentation** | Albumentations                                                                      |
| **Disease Segmentation** | **YOLOv8s-seg (Ultralytics)**                                            |
| **Insect Detection** | **YOLOv8s (Ultralytics)**                                               |
| **Questionnaire Inference** | **TabNet (PyTorch TabNet)**                                               |
| **Core ML/DL Framework** | PyTorch                                                                             |
| **Data Handling & Processing** | OpenCV, Pandas, NumPy                                                  |
| **Interactive UI (Demo)** | ipywidgets, Matplotlib, Seaborn                                    |
| **Evaluation Metrics** | Scikit-learn                                                            |

-----

## 💻 How to Run

### **Prerequisites**

  * Python \>= 3.8
  * Git
  * Docker & Docker Compose (for CVAT setup)

### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/DeepCrop-AI.git
    cd DeepCrop-AI
    ```
2.  **Create a Python virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Pre-trained Models:**
    Download the trained model files (`best.pt` for YOLO, `.zip` for TabNet) from the project's [Releases Page](https://www.google.com/search?q=https://github.com/your-username/DeepCrop-AI/releases) and place them in the specified directories.

### **Execution**

#### **Option 1: End-to-End Terminal Execution**

Follow the step-by-step procedure detailed on pages 32-33 of our submission report to run each component individually and see the final fused output.

#### **Option 2: Interactive Jupyter Notebook Demo**

1.  Launch Jupyter Notebook.
2.  Open the `multimodal_predictor.py` script (which is designed to run in a notebook environment).
3.  Run all cells.
4.  Use the `ipywidgets` file uploaders and toggle buttons to provide input and see the live multimodal prediction.

-----

## 📊 Model Performance Summary

The performance of each model component was evaluated independently.

| Model                   | Task                      | Dataset Size  | Performance Metric (Accuracy / mAP@0.5) | Avg. Inference Time (CPU) |
| ----------------------- | ------------------------- | ------------- | --------------------------------------- | ------------------------- |
| **YOLOv8s-seg** | Disease Segmentation      | 150 images    | 79% (mAP)                               | \~250ms / image            |
| **YOLOv8s** | Insect Detection          | 150 images    | 86% (mAP)                               | \~230ms / image            |
| **TabNet Disease** | Questionnaire-Based       | 150 samples   | 100% (Accuracy)                         | \~30ms / sample            |
| **TabNet Insect** | Questionnaire-Based       | 150 samples   | 100% (Accuracy)                         | \~30ms / sample            |

-----

## 👥 Team

**Team DeepCrop 1.0**
| Name                | Email                                     |
| ------------------- | ----------------------------------------- |
| Mohammed Arfath R   | mohammedarfath.r2022@vitstudent.ac.in     |
| Naresh R            | naresh.r2022a@vitstudent.ac.in            |
| Hariharan S         | hariharan.s2022d@vitstudent.ac.in         |
| Mohammad Yusuf K A  | mohammadyusuf.ka2022@vitstudent.ac.in     |

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
