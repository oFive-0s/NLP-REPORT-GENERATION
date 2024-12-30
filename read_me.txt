Accident Detection and Report Generation System

Project Overview

This project focuses on processing accident videos using advanced AI models to automatically generate detailed accident reports. The system is built with three main components:

YOLOv8 Object Detection:

Trained to detect objects, vehicles, and accidents in video footage.

Processes the input video frame by frame, identifying relevant details and generating text-based descriptions.

NLP Model for Report Generation:

Takes the textual output from the YOLOv8 model as input.

Processes this data to create a comprehensive and structured accident report.

Graphical User Interface (GUI):

Provides a user-friendly interface for interacting with the system.

Displays the generated accident report and other outputs.

Allows users to upload videos and view processing results in real-time.

Project Workflow

Input Video:

The system accepts an accident video (e.g., cr2.mp4).

YOLOv8 Model Processing:

The YOLOv8 model, trained using the weights provided in best.pt, identifies objects, vehicles, and accidents in the video.

Outputs a textual description of the detected entities and their interactions.

NLP Model Processing:

The NLP model takes the textual description from YOLOv8 and generates a detailed accident report.

This report includes information such as:

Time and location of the accident (if metadata is provided).

Number and types of vehicles involved.

Possible causes and impacts.

GUI Display:

The GUI presents the generated accident report to the user.

Displays key details and insights extracted from the video analysis.

File Structure

accident_classifier_full.pt: Backup or additional model weights.

best.pt: Trained YOLOv8 model weights for accident detection.

cr2.mp4: Sample accident video used for testing.

Requirements

To run the project, the following dependencies are required:

Python 3.8+

PyTorch

YOLOv8 framework (Ultralytics)

NLP libraries (e.g., Hugging Face Transformers, SpaCy)

OpenCV for video processing

Tkinter or PyQt for GUI development

Installation

Clone the repository.

Install dependencies using:

pip install -r requirements.txt

Place the YOLOv8 weights (best.pt) and input video (cr2.mp4) in the designated directories.

Usage

Launch the GUI to start the application:

python app.py

Use the GUI to upload a video, process it, and view the generated accident report.

Alternatively, run the scripts manually:

Process the video to generate textual descriptions:

python process_video.py --input cr2.mp4 --weights best.pt

Generate the accident report:

python generate_report.py --input descriptions.txt

The final report will be saved as accident_report.txt.

Outputs

YOLOv8 Output: Text file containing detected objects and accident descriptions.

NLP Model Output: Comprehensive accident report with structured details.

GUI Display: Visual interface showing the report and processing results.

Applications

Traffic monitoring and analysis.

Automated accident reporting for insurance claims.

Enhanced road safety and incident management.

Acknowledgments

YOLOv8 by Ultralytics for object detection.

NLP libraries for natural language processing.

Open-source community for supporting AI and video analysis tools.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For further queries or collaboration, please reach out to Abhiram Yadav M at abhiramyadavm@gmail.com.

