from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import torch  

app = Flask(__name__)

# Initialize the YOLO model
model = YOLO("model_files/best.pt")
class_names = ["healthy", "infected"]

def calculate_severity(boxes):
    """
    Calculate severity based on the size of detected areas.
    This is a simplistic example; you should adapt it based on your criteria.
    """
    total_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes) 
    print(total_area) # Assuming boxes are in [x1, y1, x2, y2] format
    if total_area > 200000:  
        return "High"
    elif total_area > 50000:
        return "Medium"
    else:
        return "Low"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file using OpenCV
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Resize the image to 640x640 using INTER_AREA interpolation
        image = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference on the resized image
        results = model([image])[0]  # Note: Adjust this line based on actual API usage
        
        classes = results.boxes.cls
        class_indices = classes.cpu().numpy().tolist() if isinstance(classes, torch.Tensor) else classes.tolist()
        class_names_list = [class_names[int(idx)] for idx in class_indices]
        
        # Process results
        boxes = results.boxes
        
        for pred in results:
            print(pred)
            pred.save(filename="result.jpg")  # save to disk
        
        # Assuming `boxes` is your Boxes object
        # Convert boxes to NumPy array and then to list
        print("classes :" , class_names_list)
        
        probs_list = None
        # Convert boxes, probs, masks, and keypoints to lists
        boxes_list = boxes.xyxy.cpu().numpy().tolist() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy.tolist()

        # Calculate severity
        

        if class_names_list[0]== 'healthy':
            return jsonify({
            'classes': 'healthy'
            }), 200
        else:
            severity = calculate_severity(boxes.xyxy)
            print(severity) 
            return jsonify({
            'severity': severity,
            'classes': 'infected plant'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
