Of course. Based on the detailed information in your hackathon submission document (`AgriThon 2.pdf`), here is a professional and comprehensive README file. It combines the structure of your draft with the specific details, code, and results from your report to create a polished and highly relevant document.

-----

# DeepCrop AI 1.0: Multimodal Crop Disease and Insect Detection

[](https://vit.ac.in/)
[](https://www.python.org/)
[](https://pytorch.org/)
[](https://opensource.org/licenses/MIT)

[cite\_start]**DeepCrop AI 1.0** is the official Round 1 submission for **AgriThon 2.0** by **Team DeepCrop 1.0**[cite: 66]. The project presents a robust multimodal pipeline for identifying sugarcane diseases and pest infestations by intelligently fusing computer vision with symptom-based textual analysis.

[cite\_start]This system was developed for the hackathon organized by the **School of Computer Science and Information Systems, VIT, Vellore**, and sponsored by the **Department of Biotechnology, Govt. of India**[cite: 59, 60, 61, 62, 63].

-----
<img width="2816" height="1536" alt="Gemini_Generated_Image_ip931sip931sip93 (1)" src="https://github.com/user-attachments/assets/aecfb77d-3e94-4bdf-b5cc-6141e3be4c0a" />

## 🏛️ System Architecture & Pipeline

[cite\_start]Our solution follows a structured, end-to-end pipeline from data annotation to a final, fused prediction[cite: 230].

1.  [cite\_start]**Annotation Phase:** The initial dataset of 50 disease and 50 insect images provided by the organizers was annotated using **CVAT (offline Docker setup)**[cite: 71, 72, 1853].
      * [cite\_start]**Crop Insects:** Annotated with **Bounding Boxes** for object detection[cite: 131].
      * [cite\_start]**Crop Diseases:** Annotated with **Segmentation Masks** for precise localization[cite: 169].
2.  [cite\_start]**Augmentation Phase:** The annotated datasets were augmented using techniques like rotation, flipping, color jittering, and contrast adjustments to expand the dataset to 150 images per class and improve model robustness[cite: 220, 238, 1850].
3.  **Model Training Phase:** Four specialized models were trained:
      * [cite\_start]**YOLOv8s-seg:** Trained on segmentation masks to identify crop diseases[cite: 227].
      * [cite\_start]**YOLOv8s:** Trained on bounding boxes to detect crop insects[cite: 227].
      * [cite\_start]**TabNet Disease Classifier:** Trained on a synthetic CSV of symptom-based Yes/No questions to predict disease presence[cite: 228, 1856].
      * [cite\_start]**TabNet Insect Classifier:** Trained similarly on insect-related symptom questions[cite: 228, 1856].
4.  [cite\_start]**Inference & Fusion Phase:** In the final step, predictions from all four models are aggregated using a strict fusion logic to deliver a final, unified diagnosis[cite: 229].

<img width="3840" height="3176" alt="DeepCrop_Architecture Diagram" src="https://github.com/user-attachments/assets/67ba95cc-d016-4c9a-84b2-6a029114cd8b" />

[cite\_start]*(This diagram is a representation of the architecture detailed on page 9 of the submission PDF [cite: 257])*

-----

## ✨ Key Features

  - **🧠 Multimodal Fusion Engine:** The core of our system. [cite\_start]A final "Present" verdict is given only if **both** the computer vision model (YOLO) and the questionnaire model (TabNet) return a positive result, minimizing false positives[cite: 995, 996, 997, 998].
  - [cite\_start]**🌿 Precise Disease Segmentation:** Uses `YOLOv8s-seg` to not just detect, but accurately outline the exact infected regions on leaves[cite: 227].
  - [cite\_start]**🐛 Robust Insect Detection:** Employs `YOLOv8s` to identify and locate insect pests with bounding boxes[cite: 227].
  - [cite\_start]**📋 Interactive Symptom-Based Diagnosis:** Leverages two `TabNet` models that guide users through a series of Yes/No questions to diagnose issues based on textual symptoms[cite: 228].
  - [cite\_start]**🧪 End-to-End Reproducibility:** The entire workflow, from CVAT setup to running terminal predictions, is fully documented for easy replication[cite: 71, 1018].
  - [cite\_start]**📊 Comprehensive Performance Metrics:** Includes detailed training graphs, confusion matrices, and performance tables for each model component[cite: 1424, 1601, 1728, 1786, 1849].

-----

## 🛠️ Tech Stack

| Component                       | Technology / Library                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| **Annotation** | **CVAT (Computer Vision Annotation Tool)**, Docker                                  |
| **Image Augmentation** | Albumentations                                                                      |
| **Disease Segmentation** | [cite\_start]**YOLOv8s-seg (Ultralytics)** [cite: 1860]                                            |
| **Insect Detection** | [cite\_start]**YOLOv8s (Ultralytics)** [cite: 1860]                                                |
| **Questionnaire Inference** | [cite\_start]**TabNet (PyTorch TabNet)** [cite: 1861]                                              |
| **Core ML/DL Framework** | PyTorch                                                                             |
| **Data Handling & Processing** | [cite\_start]OpenCV, Pandas, NumPy [cite: 1863]                                                  |
| [cite\_start]**Interactive UI (Demo)** | ipywidgets, Matplotlib, Seaborn [cite: 883, 1865]                                    |
| **Evaluation Metrics** | [cite\_start]Scikit-learn [cite: 1866]                                                           |

-----

## 💻 How to Run

### **Prerequisites**

  * [cite\_start]Python \>= 3.8 [cite: 75]
  * [cite\_start]Git [cite: 77]
  * [cite\_start]Docker & Docker Compose (for CVAT setup) [cite: 76]

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

[cite\_start]Follow the step-by-step procedure detailed on pages 32-33 of our submission report to run each component individually and see the final fused output[cite: 1018].

#### **Option 2: Interactive Jupyter Notebook Demo**

1.  Launch Jupyter Notebook.
2.  Open the `multimodal_predictor.py` script (which is designed to run in a notebook environment).
3.  Run all cells.
4.  [cite\_start]Use the `ipywidgets` file uploaders and toggle buttons to provide input and see the live multimodal prediction[cite: 1010, 1011].

-----

## 📊 Model Performance Summary

The performance of each model component was evaluated independently.

| Model                   | Task                      | Dataset Size  | Performance Metric (Accuracy / mAP@0.5) | Avg. Inference Time (CPU) |
| ----------------------- | ------------------------- | ------------- | --------------------------------------- | ------------------------- |
| **YOLOv8s-seg** | Disease Segmentation      | 150 images    | 79% (mAP)                               | \~250ms / image            |
| **YOLOv8s** | Insect Detection          | 150 images    | 86% (mAP)                               | \~230ms / image            |
| **TabNet Disease** | Questionnaire-Based       | 150 samples   | 100% (Accuracy)                         | \~30ms / sample            |
| **TabNet Insect** | Questionnaire-Based       | 150 samples   | 100% (Accuracy)                         | \~30ms / sample            |
[cite\_start]*(Table adapted from the performance summary on page 47 [cite: 1849, 1850])*

-----

## 👥 Team

**Team DeepCrop 1.0**
| Name                | Email                                     |
| ------------------- | ----------------------------------------- |
| Mohammed Arfath R   | mohammedarfath.r2022@vitstudent.ac.in     |
| Naresh R            | naresh.r2022a@vitstudent.ac.in            |
| Hariharan S         | hariharan.s2022d@vitstudent.ac.in         |
| Mohammad Yusuf K A  | mohammadyusuf.ka2022@vitstudent.ac.in     |
[cite\_start]*(Team member information from page 1 of the submission [cite: 68])*

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
