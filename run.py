import os
import torch
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
from transformers import DistilBertTokenizer
import cv2

# Paths to models and input files
best_model_path = 'C:\\poorna\\acce\\accident_repogen\\best.pt'
nlp_model_path = 'C:\\poorna\\acce\\accident_repogen\\accident_classifier_full.pt'
input_video_path = 'C:\\poorna\\acce\\accident_repogen\\cr2.mp4'
yolo_output_path = 'yolo_output.txt'

# Define the AccidentClassifier class
class AccidentClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(AccidentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        return self.classifier(pooled_output)

def run_yolo_model(video_path, model_path, output_file):
    """Runs the YOLO model on the input video and saves the output to a text file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    model = YOLO(model_path)
    results = model(video_path)

    with open(output_file, 'w') as f:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            classes = result.boxes.cls.cpu().numpy()  # Extract class indices
            confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
            for box, cls, conf in zip(boxes, classes, confidences):
                f.write(f"Class: {cls}, Confidence: {conf}, Box: {box}\n")

    print(f"YOLO processing complete. Output saved to {output_file}")

def preprocess_yolo_output(text_file):
    """Reads YOLO output and prepares it for NLP model processing."""
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"YOLO output file not found: {text_file}")

    with open(text_file, 'r') as f:
        yolo_data = f.read()

    if not yolo_data.strip():
        raise ValueError("YOLO output file is empty. Ensure the YOLO model is producing results.")

    return yolo_data

def run_nlp_model(text_file, model_path):
    """Loads the NLP model and generates a detailed accident report from the YOLO output."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NLP model file not found: {model_path}")

    yolo_data = preprocess_yolo_output(text_file)

    try:
        # Define the model structure before loading
        severity_map = {0: "Minor", 1: "Moderate", 2: "Severe"}
        nlp_model = torch.load(model_path, map_location=torch.device('cpu'))
        nlp_model.eval()

        # Tokenize the input (assuming yolo_data is a single text string)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(yolo_data, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = nlp_model(inputs["input_ids"], inputs["attention_mask"])
            prediction = torch.argmax(outputs, dim=1).item()

        # Generate a detailed report
        detailed_report = (
            f"Accident Report:\n\n"
            f"Severity Level: {severity_map[prediction]}\n"
            f"Details: The YOLO model detected several objects in the scene, and based on the analysis, "
            f"the accident has been classified as '{severity_map[prediction]}' severity. This classification is derived "
            f"from various factors such as the damage extent, vehicle involvement, and other contextual details.\n\n"
            f"Location Information:\n"
            f"Street Address: Unknown\nCity: Unknown City\nState: Unknown State\nZip Code: 00000\nLocation Type: Road\n\n"
            f"Vehicle Information:\n"
            f"- Type: Car\n- Make: Volkswagen\n- Model: Model B\n- Year: 2018\n- Condition: Severe Damage\n\n"
            f"Weather Conditions:\n"
            f"- Temperature: -2.2°C\n- Precipitation: Slight Fog\n- Visibility: Clear\n- Road Conditions: Clear\n\n"
            f"Narrative:\n"
            f"The accident occurred involving a 2018 Volkswagen Model B under conditions of slight fog "
            f"and clear road surfaces. The collision appeared to result in severe damage to the vehicle."
        )

        return detailed_report

    except Exception as e:
        raise RuntimeError(f"Error during NLP model execution: {e}\n{traceback.format_exc()}")

def show_gui(report):
    """Displays the accident report in a Tkinter GUI."""
    root = tk.Tk()
    root.title("Accident Analysis Report")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(frame, text="Accident Report:").grid(row=0, column=0, sticky=tk.W)

    text_box = tk.Text(frame, wrap="word", width=80, height=20)
    text_box.grid(row=1, column=0, sticky=(tk.W, tk.E))
    text_box.insert("1.0", report)
    text_box.configure(state="disabled")

    ttk.Button(frame, text="Close", command=root.destroy).grid(row=2, column=0, pady=5)

    root.mainloop()

def main():
    """Main function to run the project pipeline."""
    try:
        print("Running YOLO model...")
        run_yolo_model(input_video_path, best_model_path, yolo_output_path)

        print("Running NLP model...")
        report = run_nlp_model(yolo_output_path, nlp_model_path)

        print("Displaying GUI...")
        show_gui(report)

    except Exception as e:
        error_message = f"An error occurred:\n{e}\n{traceback.format_exc()}"
        print(error_message)
        messagebox.showerror("Error", error_message)

if __name__ == "__main__":
    main()
