# Face-Recognition-Model
A face recognition application developed in Python using the DeepFace framework and the FaceNet512 model. The system identifies individuals by comparing facial embeddings extracted from an input image against a structured image database.


Tech Stack

- Python 3.11
- DeepFace
- tf-keras
- NumPy

Project Structure

FaceRecognition/
│
├── dataset/
│   ├── Person_1/
│   ├── Person_2/
│   ├── Person_3/
│   └── ...
│
├── test/
│   └── test.jpg
│
└── main.py

Installation

Clone the repository:
git clone https://github.com/your-username/FaceRecognition.git
cd FaceRecognition

Install the required packages:
pip install deepface tensorflow tf-keras numpy

 Test Image:
Place the image to be recognized inside the `test` directory and name it:
test.jpg




Run:
python main.py

 Sample Output:
===== FACE RECOGNITION RESULT =====

Recognized Person : Person_1
Confidence        : 94.53%

Workflow:
1. Load the image database.
2. Detect faces from each image.
3. Generate FaceNet512 embeddings.
4. Cache embeddings for faster future execution.
5. Load the input image.
6. Generate its facial embedding.
7. Compare it against the stored embeddings.
8. Return the closest match with its confidence score.

Notes:
- During the first execution, the FaceNet512 model is downloaded automatically.
- The first run also generates facial embeddings for the entire dataset, which may take several minutes depending on the number of images.
- Subsequent executions reuse the cached embeddings, significantly reducing execution time.

Dependencies:
deepface
tf-keras
Numpy

Dataset: [Hollywood Celebrity Facial Recognotion Dataset]((https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset?))
