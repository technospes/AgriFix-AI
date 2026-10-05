from ultralytics import YOLO
print("Loading model...")
model = YOLO("yolov8n.pt")
print("Exporting to ONNX for Unity Sentis...")
model.export(format="onnx", opset=15)
print("✅ Success! Check your folder for 'yolov8n.onnx'")