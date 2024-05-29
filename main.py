from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import torch 

app = Flask(__name__)

# Initialize the YOLO model
model = YOLO("model_files/best.pt")

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
        
        # Process results
        boxes = results.boxes
        masks = results.masks
        keypoints = results.keypoints
        probs = results.probs
        obb = results.obb
        
        # Assuming `boxes` is your Boxes object
        # Convert boxes to NumPy array and then to list
        boxes_np = None
        
        # Convert boxes, probs, masks, and keypoints to lists
        boxes_list = boxes.xyxy.cpu().numpy().tolist() if isinstance(boxes.xyxy, torch.Tensor) else boxes.xyxy.tolist()
        probs_list = probs.cpu().numpy().tolist() if isinstance(probs, torch.Tensor) else probs.tolist()
        
        # Assuming masks and keypoints are tensors or arrays; adjust accordingly
        #masks_list = masks.cpu().numpy().tolist() if isinstance(masks, torch.Tensor) else masks.tolist()
        #keypoints_list = keypoints.cpu().numpy().tolist() if isinstance(keypoints, torch.Tensor) else keypoints.tolist()
        
        # Example: Return the bounding boxes, probs, masks, and keypoints as JSON
        return jsonify({
            'boxes': boxes_list,
            'probs': probs_list
            #'masks': masks_list,
            #'keypoints': keypoints_list
        }), 200

        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)
