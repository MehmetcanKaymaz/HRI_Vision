import cv2
import torch
import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image

# Load the pre-trained ResNet-18 model
model = torch.load('models/model5.pth')

model.to('cpu')

# Set the model to evaluation mode
model.eval()

# Define the data transformations to be applied to the images
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open a connection to the default webcam
cap = cv2.VideoCapture(0)

# Loop over the frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Apply the data transformations to the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(np.uint8(img))

    # Apply the data transformations to the image
    img = data_transforms(img).unsqueeze(0)


    # Make a prediction with the model
    with torch.no_grad():
        output = model(img)

    # Get the predicted class label
    _, predicted = torch.max(output.data, 1)
    label = predicted.item()

    # Display the predicted class label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(frame, 'Predicted Class: ' + str(label), org, font,
                fontScale, color, thickness, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
