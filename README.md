# 🤖A Deep Learning Framework for Environment Classification, Object Centroid Detection, and Collision-Free Robotic Operation   
**Internship Project – SHOBOT, IIT Madras**  
**Author:** Anamika Panickar  
**Duration:** 16 June 2025 – 16 September 2025  

---

## 🧭 Introduction  

This project, developed during my internship at **SHOBOT, IIT Madras**, focuses on building an **AI-powered robotic navigation system** that perceives and interacts with its environment using **Computer Vision (CV)** and **Deep Learning (DL)**.  

The system integrates three core tasks:  
1. **Scene Classification** using ResNet18  
2. **Object Detection & Depth Estimation** using YOLOv8 and MiDaS  
3. **Collision Avoidance & Navigation Integration** combining all modules  

Together, these modules enable an intelligent robot to analyze its environment, detect obstacles, estimate distances, and plan safe paths dynamically.

---

## ⚙️ Tech Stack  

| Category | Tools/Frameworks |
|-----------|------------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Models | YOLOv8, MiDaS, ResNet18 |
| Dataset Tool | Roboflow |
| Others | NumPy, torchvision |

---

## 📁 Folder Structure
```

INTERNSHIP/  
│  
├── training\_data/  
│ ├── train/  
│ └── valid/  
│  
├── venv/  
│  
├── centroid.py  
├── cv\_model.py  
├── depth\_midas.py  
├── image\_scene\_prediction.py  
├── object\_detection\_with\_scene\_classification\_image.py  
├── video\_object\_detection.py  
├── video\_scene\_prediction.py  
│  
├── resnet18\_scene\_classifier.py  
├── yolov8n.pt  
├── yolov8s.pt  
├── yolov8m.pt  
├── yolov8l.pt  
│  
├── midas\_v21\_small\_256.pt  
│  
├── sample\_image1.jpg  
├── sample\_video1.mp4  
├── sample\_video2.mp4  
│  
├── out.jpg  
├── requirements.txt  
└── README.md

```yaml
---

## 🧩 Setup  

### 1️⃣ Clone and Install  
```bash
git clone https://github.com/<your-username>/AI-Robotic-Navigation.git
cd AI-Robotic-Navigation
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Using Roboflow for Dataset Creation

Roboflow was used for creating and labeling datasets for **scene classification** (ResNet18) and **object detection** (YOLOv8).

### 🔹 Steps:

1.  **Create Project** → https://roboflow.com
    
    -   Type: “Classification” for scene classification
        
    -   Type: “Object Detection” for YOLOv8
        
2.  **Upload Images** for each environment:
    
    -   `Confined_Space`, `Open_Space`, `Non_Complex_Environment`
        
3.  **Label Images** – assign class labels or bounding boxes.
    
4.  **Preprocess & Augment** – resize, flip, rotate, adjust brightness.
    
5.  **Generate Dataset Version** and **export** as:
    
    -   `YOLOv8` for detection
        
    -   `Classification` for ResNet18
        
6.  **Download via Python API**
    

```python
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("robot-environment-scenes")
dataset = project.version(1).download("yolov8")  # or "classification"
```

---

# 🧩 TASK 1 – Scene Classification (ResNet18)

### 🎯 Objective

Classify the robot’s environment into three categories:

-   **Confined Space**
    
-   **Open Space**
    
-   **Non-complex Environment**
    

### ⚙️ Model

-   **Architecture:** ResNet18
    
-   **Dataset:** Created in Roboflow
    
-   **Input:** Single RGB image
    
-   **Output:** Environment class label
    

### ▶️ How to Run

```bash
python image_scene_prediction.py
```

### 📊 Outcome
    
-   Model helps robot adjust navigation based on environment type
    

---

# 🧩 TASK 2 – Object Detection & Depth Estimation (YOLOv8 + MiDaS)

### 🎯 Objective

Detect and segment objects in real time and estimate their distances for navigation safety.

### ⚙️ Models Used

-   **YOLOv8** – Object detection and segmentation
    
-   **MiDaS** – Monocular depth estimation
    

### ▶️ How to Run

-   **YOLOv8 Detection:**
    
    ```bash
    python video_object_detection.py
    ```
    
-   **Depth Estimation:**
    
    ```bash
    python depth_midas.py
    ```
    
-   **Combined YOLO + MiDaS:**
    
    ```bash
    python object_detection_with_scene_classification_image.py
    ```
    

### 📊 Output

-   Bounding boxes and masks around detected objects
    
-   Depth map overlay for obstacle proximity
    
-   Saved image: `out.jpg`
    

---

# 🧩 TASK 3 – Collision Avoidance & Navigation Integration

### 🎯 Objective

Integrate scene classification, object detection, and depth estimation into one intelligent navigation system.

### ⚙️ Logic Flow

1.  **Scene Classification** (ResNet18) → Identify environment type.
    
2.  **Object Detection & Depth Estimation** (YOLO + MiDaS) → Detect and measure distances.
    
3.  **Decision Layer:**
    
    -   If confined → Use **PRM** for safe pathfinding
        
    -   If open → Use **RRT\*** for efficient navigation
        

### ▶️ Integration Script

```bash
python object_detection_with_scene_classification_image.py
```

### 📊 Outcome

-   Dynamic path planning
    
-   Obstacle detection and avoidance in real time
    
-   Improved safety and efficiency of navigation
    

---

## 🧪 Results Summary

| Module | Model Used | Performance |
| --- | --- | --- |
| Scene Classification | ResNet18 | 94% Accuracy |
| Object Detection | YOLOv8s | Real-time (30–40 FPS) |
| Depth Estimation | MiDaS | Reliable relative depth |
| Integration | YOLO + MiDaS + ResNet | Smart navigation decisions |

---

## 🧠 Key Learnings

-   Integration of multiple **deep learning models** in one robotic pipeline.
    
-   Dataset creation and augmentation using **Roboflow**.
    
-   Strong hands-on experience in **object detection**, **depth estimation**, and **scene analysis**.
    
-   Applied AI for **autonomous navigation and obstacle avoidance**.
    


## 📜 Conclusion

This internship at **SHOBOT, IIT Madras** allowed me to combine **computer vision**, **deep learning**, and **robotics** into one cohesive project.  
By integrating scene understanding, depth perception, and dynamic navigation, the system represents a step toward **fully autonomous, environment-aware robots**.