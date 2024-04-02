import tkinter as tk
from tkinter import ttk, filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Precaution messages for different severities
precautions = {
    "Low": "Take healthy foods and avoid common foods causing PCOS.\nRegular exercise is recommended.",
    "Medium": "Consult a doctor for personalized treatment plans.\nMaintain a healthy lifestyle with diet and exercise.",
    "High": "Seek medical attention immediately.\nFollow doctor's recommendations strictly.\nManage stress and prioritize healthy habits."
}

model = load_model('pcos_detection_model.h5')

# Global variables for form submission
probability = 0
weights = [2, 1, 3, 1, 2, 2, 1, 1, 2]
total_weight = sum(weights)

# Function to calculate PCOS probability
def calculate_probability():
    global probability
    probability = 0
    responses = [q1_var.get(), q2_var.get(), q3_var.get(), q4_var.get(), q5_var.get(), q6_var.get(), q7_var.get(), q8_var.get(), q9_var.get()]
    for i, response in enumerate(responses):
        if response == "Yes":
            probability += weights[i]
    pcos_probability = (probability / total_weight) * 100
    result_label.config(text=f"Probability of PCOS: {pcos_probability:.2f}%", foreground="white")

# Function to predict PCOS severity
def predict_pcos_severity(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    prediction = model.predict(img_array)
    severity = 'Low' if prediction > 0.33 else 'Medium' if prediction > 0.66 else 'High'
    return severity

# Function to handle image selection
def choose_image():
    img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if img_path:  # If user selected a file
        img_label.config(text=img_path)
        severity = predict_pcos_severity(img_path)
        severity_label.config(text=f"Severity: {severity}")
        precaution_label.config(text=precautions[severity])

# Create main window
root = tk.Tk()
root.title("PCOS Assistant")

# Create tab control
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab1, text="PCOS Probability Calculator")
tab_control.add(tab2, text="PCOS Severity Prediction")
tab_control.pack(expand=1, fill="both")

# Tab 1: PCOS Probability Calculator
q1_var = tk.StringVar()
q2_var = tk.StringVar()
q3_var = tk.StringVar()
q4_var = tk.StringVar()
q5_var = tk.StringVar()
q6_var = tk.StringVar()
q7_var = tk.StringVar()
q8_var = tk.StringVar()
q9_var = tk.StringVar()

questions = [
    "Do you experience irregular periods?",
    "Do you notice abnormal weight gain in the lower body?",
    "Do you have recurring stomach ulcers?",
    "Do you experience any skin tags or recurring rashes?",
    "Do you have male-patterned baldness?",
    "Do you notice increased facial hair (hirsutism)?",
    "Do you struggle with losing weight?",
    "Do you experience acne or oily skin?",
    "Do you face difficulties getting pregnant?"
]

for i, question in enumerate(questions):
    label = ttk.Label(tab1, text=question)
    label.grid(row=i, column=0, padx=14, pady=5, sticky="w")

    yes_radio = ttk.Radiobutton(tab1, text="Yes", variable=eval(f"q{i+1}_var"), value="Yes")
    yes_radio.grid(row=i, column=1, padx=14, pady=5, sticky="w")

    no_radio = ttk.Radiobutton(tab1, text="No", variable=eval(f"q{i+1}_var"), value="No")
    no_radio.grid(row=i, column=2, padx=14, pady=5, sticky="w")

calculate_button = ttk.Button(tab1, text="Calculate Probability", command=calculate_probability)
calculate_button.grid(row=len(questions), columnspan=3, padx=14, pady=14)

result_label = ttk.Label(tab1, text="", foreground="white")
result_label.grid(row=len(questions) + 1, columnspan=3, padx=14, pady=5)

for i in range(len(questions)):
    tab1.grid_rowconfigure(i, weight=1)
    tab1.grid_columnconfigure(0, weight=1)
    tab1.grid_columnconfigure(1, weight=1)
    tab1.grid_columnconfigure(2, weight=1)

# Tab 2: PCOS Severity Prediction
image_frame = tk.Frame(tab2, width=300, height=250, bd=1, relief=tk.SUNKEN, padx=10, pady=10)
image_frame.pack()

img_label = tk.Label(image_frame, text="No image selected", font=("Arial", 12))
img_label.pack()

button_frame = tk.Frame(tab2)
button_frame.pack(padx=10, pady=10)

choose_btn = tk.Button(button_frame, text="Select Image", command=choose_image)
choose_btn.pack(side=tk.LEFT, padx=10)

clear_btn = tk.Button(button_frame, text="Clear", command=lambda: [img_label.config(
    text="No image selected", font=("Arial", 12)), severity_label.config(text="Severity:"), precaution_label.config(text="")])
clear_btn.pack(side=tk.RIGHT)

severity_label = tk.Label(tab2, text="Severity:", font=("Arial", 12))
severity_label.pack(padx=10, pady=10)

precaution_label = tk.Label(tab2, text="", font=("Arial", 12))
precaution_label.pack(padx=10, pady=10, fill=tk.X)

root.mainloop()
