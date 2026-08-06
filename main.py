from deepface import DeepFace
import os

# Paths
dataset_path = "dataset"
test_image = "test/test.jpg"

# Find closest match
result = DeepFace.find(
    img_path=test_image,
    db_path=dataset_path,
    model_name="Facenet512",
    detector_backend="opencv",
    enforce_detection=False
)

if len(result[0]) > 0:

    match = result[0].iloc[0]

    # Full matched image path
    matched_path = match["identity"]

    # Celebrity name (folder name)
    celebrity = os.path.basename(os.path.dirname(matched_path))

    # Distance
    distance = float(match["distance"])

    # Simple confidence calculation
    confidence = max(0, (1 - distance)) * 100

    print("\n===== FACE RECOGNITION RESULT =====")
    print(f"Recognized Person : {celebrity}")
    print(f"Confidence        : {confidence:.2f}%")

else:
    print("Unknown Person")