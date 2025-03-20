import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 1: Simulate medical image data
# Let's pretend each image is represented by 256 pixel values (features)
# Generate data for 100 simulated images
X = np.random.rand(100, 256)  # 100 images, each with 256 random pixel values

# Step 2: Generate labels for the data
# 0 = No tumor, 1 = Tumor
y = np.random.randint(1, 2, 100)  # Randomly assign 0 or 1 to each image

# Step 3: Train the Random Forest model on the simulated data
model = RandomForestClassifier()  # Initialize the model
model.fit(X, y)  # Train the model with our data

# Step 4: Simulate a new medical image
new_image = np.random.rand(10, 256)  # Ten new images with 256 pixel values

# Step 5: Use the trained model to predict on the new image
prediction = model.predict(new_image)

# Step 6: Print the result based on the prediction
if prediction[0] == 1:
    print("Tumor detected!")
else:
    print("No tumor detected.")
