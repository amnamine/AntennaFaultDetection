import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np

# Define the exact features used during training
FEATURES = [
    'Length', 'Width', 'Height', 'Permittivity', 'Conductivity', 
    'Bend', 'Feed', 'S11', 'VSWR', 'Gain', 
    'Efficiency', 'Bandwidth', 'epsilon_r'
]

# Map dropdown options to their corresponding saved .pkl files
MODEL_FILES = {
    'WiFi Fault': 'model_WiFi_Fault.pkl',
    'BT Fault': 'model_BT_Fault.pkl',
    'WiFi Status': 'model_WiFi_Status.pkl',
    'BT Status': 'model_BT_Status.pkl'
}

# The exact class names extracted from antenna_fault.csv
# LabelEncoder sorts classes alphabetically, so the index matches the integer prediction
CLASS_MAPPINGS = {
    'WiFi Fault': [
        'Bending', 'Body_Effect', 'Conductivity_Degradation', 
        'Cracks', 'Humidity_Sweat', 'No_Fault', 
        'Rupture_Coupure', 'Strong_Flexion'
    ],
    'BT Fault': [
        'Bending', 'Body_Effect', 'Conductivity_Degradation', 
        'Coupure', 'Cracks', 'Humidity_or_Sweat', 
        'No_Fault', 'Rupture', 'Strong_Flexion'
    ],
    'WiFi Status': ['Fault', 'Normal'],
    'BT Status': ['Fault', 'Normal']
}

# --- UI Styling Constants ---
BG_COLOR = "#1e272e"          # Dark background
FG_COLOR = "#d2dae2"          # Light text
ACCENT_COLOR = "#0fbcf9"      # Bright blue accent
BTN_COLOR = "#3c40c6"         # Button color
BTN_HOVER = "#575fcf"         # Button hover color
FONT_TITLE = ("Helvetica", 18, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_ENTRY = ("Helvetica", 11)
FONT_RESULT = ("Helvetica", 14, "bold")

class AntennaPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Antenna Performance Predictor")
        self.root.geometry("650x650")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.entries = {}
        self.build_ui()

    def build_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Antenna Performance & Fault Predictor", 
                               font=FONT_TITLE, bg=BG_COLOR, fg=ACCENT_COLOR)
        title_label.pack(pady=20)

        # Model Selection Frame
        model_frame = tk.Frame(self.root, bg=BG_COLOR)
        model_frame.pack(fill="x", padx=40, pady=10)
        
        tk.Label(model_frame, text="Select Target Model:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR).pack(side="left", padx=10)
        
        self.model_var = tk.StringVar(value="WiFi Fault")
        
        # Style the combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground=BG_COLOR, background=BTN_COLOR, foreground="white")
        
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                      values=list(MODEL_FILES.keys()), state="readonly", width=25, font=FONT_LABEL)
        model_dropdown.pack(side="left", padx=10)

        # Inputs Frame (2 columns)
        input_frame = tk.Frame(self.root, bg=BG_COLOR)
        input_frame.pack(fill="both", expand=True, padx=40, pady=10)

        # Generate entry fields dynamically
        for i, feature in enumerate(FEATURES):
            row = i // 2
            col = (i % 2) * 2
            
            # Label
            lbl = tk.Label(input_frame, text=f"{feature}:", font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR, anchor="e", width=15)
            lbl.grid(row=row, column=col, padx=(0, 5), pady=10, sticky="e")
            
            # Entry
            ent = tk.Entry(input_frame, font=FONT_ENTRY, bg="#485460", fg="white", insertbackground="white", width=15, relief="flat")
            ent.grid(row=row, column=col+1, padx=(0, 20), pady=10, sticky="w")
            
            # Default value (0.0) for easier testing
            ent.insert(0, "0.0")
            self.entries[feature] = ent

        # Predict Button
        self.predict_btn = tk.Button(self.root, text="PREDICT", font=("Helvetica", 12, "bold"), 
                                     bg=BTN_COLOR, fg="white", activebackground=BTN_HOVER, activeforeground="white",
                                     relief="flat", cursor="hand2", command=self.run_prediction, width=20)
        self.predict_btn.pack(pady=20)

        # Result Label
        self.result_label = tk.Label(self.root, text="Result will appear here", 
                                     font=FONT_RESULT, bg=BG_COLOR, fg="#0be881")
        self.result_label.pack(pady=10)

    def run_prediction(self):
        target = self.model_var.get()
        model_file = MODEL_FILES[target]

        # 1. Gather Inputs
        input_data = {}
        for feature in FEATURES:
            val_str = self.entries[feature].get()
            try:
                input_data[feature] = [float(val_str)]
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {feature}. Please enter a number.")
                return

        # 2. Convert to DataFrame (so feature names match training data)
        df_input = pd.DataFrame(input_data)

        # 3. Load Model and Predict
        try:
            model = joblib.load(model_file)
            prediction_int = model.predict(df_input)[0]
            
            # 4. Map the integer prediction to the actual class name
            predicted_class_name = CLASS_MAPPINGS[target][prediction_int]
            
            # Change color based on if it's a fault or normal/no fault (Optional visual cue)
            text_color = "#0be881" if "Normal" in predicted_class_name or "No_Fault" in predicted_class_name else "#ff3f34"
            
            self.result_label.config(text=f"Predicted {target}:\n{predicted_class_name}", fg=text_color)
            
        except FileNotFoundError:
            messagebox.showerror("Model Not Found", f"Could not find {model_file}.\nPlease ensure it is in the same directory as this script.")
        except IndexError:
             messagebox.showerror("Mapping Error", f"The predicted index {prediction_int} is out of bounds for the class mapping.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AntennaPredictorApp(root)
    root.mainloop()