from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')


model.train(data="datasets\data.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("1.png")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
