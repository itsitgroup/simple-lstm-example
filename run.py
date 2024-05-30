import subprocess

# Step 1: Install the required dependencies
print("Installing dependencies...")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Step 2: Run the preprocessing script to prepare the data
print("Running preprocessing script...")
subprocess.run(["python", "preprocessing.py"])

# Step 3: Define and compile the model
print("Running model definition script...")
subprocess.run(["python", "model.py"])

# Step 4: Train the model
print("Running training script...")
subprocess.run(["python", "train.py"])

# Step 5: Evaluate the model
print("Running evaluation script...")
subprocess.run(["python", "evaluate.py"])
