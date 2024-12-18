import cv2
import numpy as np
import os
from tkinter import Tk, Button, Label, Entry, StringVar, messagebox, Toplevel
from datetime import datetime
import csv
from PIL import Image, ImageTk
from datetime import date
import mysql.connector
import pandas as pd

#Database connection
conn = mysql.connector.connect(host='localhost', password='datascience2025', user = 'root')
if conn.is_connected():
    print(" Database Connection established....")

curobj = conn.cursor(buffered = True)
curobj.execute('use litdb;')



# Paths
DATASET_PATH = "face_datasets"
MODEL_PATH = "face_trainer.yml"
USER_DATA_FILE = "user_data.csv"
ATTENDANCE_FILE = "attendance.csv"

# Ensure directories and files exist
os.makedirs(DATASET_PATH, exist_ok=True)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Roll No", "Academic Year", "Branch", "Gender", "User ID"])

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "Name", "Roll No", "Date", "Time", "Status"])

# Haar Cascade and Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Function to capture and save face data
def capture_faces():
    global captured_id
    if not os.path.exists(MODEL_PATH):
        recognizer.read(MODEL_PATH)
        print("Model not trained. Proceeding without validation.")

    # Load existing user data for comparison
    user_data = {}
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                user_data[int(row[5])] = row  # User ID as key

    captured_id = len(user_data) + 1  # Generate a new User ID if needed
    current_date = date.today().strftime("%Y-%m-%d")
    date_folder_path = os.path.join(DATASET_PATH, current_date)
    os.makedirs(date_folder_path, exist_ok=True)

    user_folder_path = os.path.join(date_folder_path, f"user_{captured_id}")
    os.makedirs(user_folder_path, exist_ok=True)

    camera = cv2.VideoCapture(0)
    count = 0
    print("Press 'Q' to stop capturing.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error accessing the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            # Check if the face is already registered
            if os.path.exists(MODEL_PATH):
                recognizer.read(MODEL_PATH)
                user_id, confidence = recognizer.predict(face)
                if confidence < 50:  # Recognized with high confidence
                    if user_id in user_data:
                        user_info = user_data[user_id]
                        messagebox.showinfo(
                            "Info",
                            f"User already exists: {user_info[0]} (Roll No: {user_info[1]}). "
                            f"Please mark attendance instead."
                        )
                        camera.release()
                        cv2.destroyAllWindows()
                        return

            # Save new face data
            count += 1
            file_path = os.path.join(user_folder_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capturing Faces", frame)
        if count >= 50 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    if count > 0:
        print(f"Captured {count} images. Redirecting to Registration.")
        open_registration_ui()
    else:
        print("No images captured. Try again.")



# Function to open registration UI
def open_registration_ui():
    def register_user():
        name = name_var.get()
        roll_no = roll_no_var.get()
        year = year_var.get()
        branch = branch_var.get()
        gender = gender_var.get()

        if not all([name, roll_no, year, branch, gender]):
            messagebox.showerror("Error", "All fields are required.")
            return

        # Save user details
        with open(USER_DATA_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, roll_no, year, branch, gender, captured_id])

        messagebox.showinfo("Success", "User Registered Successfully.")
        reg_window.destroy()

    reg_window = Toplevel()
    reg_window.title("User Registration")
    reg_window.geometry("400x400")

    Label(reg_window, text="Registration Form", font=("Helvetica", 16)).pack(pady=10)

    name_var, roll_no_var, year_var, branch_var, gender_var = StringVar(), StringVar(), StringVar(), StringVar(), StringVar()

    Label(reg_window, text="Name").pack()
    Entry(reg_window, textvariable=name_var).pack()

    Label(reg_window, text="Roll No").pack()
    Entry(reg_window, textvariable=roll_no_var).pack()

    Label(reg_window, text="Academic Year").pack()
    Entry(reg_window, textvariable=year_var).pack()

    Label(reg_window, text="Branch").pack()
    Entry(reg_window, textvariable=branch_var).pack()

    Label(reg_window, text="Gender").pack()
    Entry(reg_window, textvariable=gender_var).pack()

    Button(reg_window, text="Register", command=register_user).pack(pady=20)


# Function to train the recognizer
def train_model():
    faces, ids = [], []
    for date_folder_name in os.listdir(DATASET_PATH):
        date_folder_path = os.path.join(DATASET_PATH, date_folder_name)
        if os.path.isdir(date_folder_path):
            for folder_name in os.listdir(date_folder_path):
                user_folder_path = os.path.join(date_folder_path, folder_name)
                if os.path.isdir(user_folder_path):
                    for image_name in os.listdir(user_folder_path):
                        img_path = os.path.join(user_folder_path, image_name)
                        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(gray_img)
                        ids.append(int(folder_name.split("_")[1]))  # Extract User ID

    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_PATH)
    messagebox.showinfo("Success", "Model Trained Successfully.")


# Function to mark attendance
def mark_attendance():
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", "Model not trained. Train the model first.")
        return

    recognizer.read(MODEL_PATH)
    user_data = {}
    
    # Load user data from the CSV file
    with open(USER_DATA_FILE, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            user_data[int(row[5])] = row  # User ID as key

    # Read existing attendance to prevent duplicate marking
    existing_attendance = {}
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                user_id, date_recorded = row[0], row[3]
                existing_attendance[(user_id, date_recorded)] = True

    camera = cv2.VideoCapture(0)
    print("Press 'Q' to stop attendance.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error accessing the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            user_id, confidence = recognizer.predict(face)
            if confidence < 50:  # Lower confidence means better match
                user_info = user_data.get(user_id)
                if user_info:
                    now = datetime.now()
                    date_today = now.strftime("%Y-%m-%d")
                    time_now = now.strftime("%H:%M:%S")

                    if (str(user_id), date_today) not in existing_attendance:
                        # Append attendance if not already marked for today
                        with open(ATTENDANCE_FILE, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([user_id, user_info[0], user_info[1], date_today, time_now, "Present"])
                        
                        existing_attendance[(str(user_id), date_today)] = True  # Update the in-memory record
                        cv2.putText(frame, f"{user_info[0]}: Marked Present", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        print(f"Attendance marked for {user_info[0]} on {date_today}.")
                    else:
                        cv2.putText(frame, f"{user_info[0]}: Already Marked", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        print(f"Attendance already marked for {user_info[0]} on {date_today}.")
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Marking Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


# GUI Application
def start_gui():
    def play_video():
        # Read video frames
        cap = cv2.VideoCapture("C:/Users/Rudra PC/Videos/Screen Recordings/face.mp4")  # Replace with the path to your video file

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # Convert the frame to RGB (Tkinter requires RGB format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1000, 700))  # Resize to fit the window
                img = ImageTk.PhotoImage(image=Image.fromarray(frame))

                # Update the label with the new frame
                video_label.configure(image=img)
                video_label.image = img

                # Schedule the next frame update
                video_label.after(10, update_frame)
            else:
                cap.release()

        update_frame()

    root = Tk()
    root.title("Face Attendance System")
    root.geometry("1000x700")
    root.configure(bg="#ADD8E6")

    # Video background
    video_label = Label(root)
    video_label.place(x=0, y=0, width=1000, height=700)

    # Start video playback
    play_video()

    # Overlay buttons on the video using absolute positioning
    Button(root, text="Capture Face", command=capture_faces, width=20, font=("Helvetica", 12)).place(x=150, y=600)
    Button(root, text="Train Model", command=train_model, width=20, font=("Helvetica", 12)).place(x=400, y=600)
    Button(root, text="Mark Attendance", command=mark_attendance, width=20, font=("Helvetica", 12)).place(x=650, y=600)

    # Title Label (overlay on video)
    Label(
        root, text="Face Attendance System", font=("Helvetica", 16, "bold"), bg="#ADD8E6"
    ).place(x=400, y=50)

    root.mainloop()


'''
def start_gui():
    def play_video():
        # Read video frames
        cap = cv2.VideoCapture("C:/Users/Rudra PC/Videos/Screen Recordings/face.mp4")  # Replace with the path to your video file

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # Convert the frame to RGB (Tkinter requires RGB format)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1000, 700))  # Resize to fit the window
                img = ImageTk.PhotoImage(image=Image.fromarray(frame))

                # Update the label with the new frame
                video_label.configure(image=img)
                video_label.image = img

                # Schedule the next frame update
                video_label.after(10, update_frame)
            else:
                cap.release()

        update_frame()

    root = Tk()
    root.title("Face Attendance System")
    root.geometry("1000x1000")
    root.configure(bg="#ADD8E6")

    # Video background
    video_label = Label(root)
    video_label.pack()

    # Start video playback
    play_video()

    # Title Label (overlay on video)
    Label(
        root, text="Face Attendance System", font=("Helvetica", 16, "bold"), bg="#ADD8E6"
    ).pack(pady=10)

    # Buttons
    Button(root, text="Capture Face", command=capture_faces, width=20).pack(pady=10)
    Button(root, text="Train Model", command=train_model, width=20).pack(pady=10)
    Button(root, text="Mark Attendance", command=mark_attendance, width=20).pack(pady=10)

    root.mainloop()
    '''
'''
def start_gui():
    root = Tk()        
    root.title("Face Attendance System")
    root.geometry("400x400")
    root.configure(bg="#ADD8E6")

    #Image for background
    image_path = "your_image_path.jpg"  # Replace with the path to your image file
    try:
        img = Image.open(image_path)
        img = img.resize((150, 150), Image.ANTIALIAS)  # Resize image to fit the GUI
        photo = ImageTk.PhotoImage(img)
        image_label = Label(root, image=photo, bg="#ADD8E6")  # Add the image to a label
        image_label.image = photo  # Keep a reference to prevent garbage collection
        image_label.pack(pady=10)  # Add padding around the image
    except Exception as e:
        print(f"Error loading image: {e}")
        Label(root, text="Image not found.", bg="#ADD8E6", fg="red").pack(pady=10)

    Label(root, text="Face Attendance System", font=("Helvetica", 16)).pack(pady=20)

    Button(root, text="Capture Face", command=capture_faces, width=20).pack(pady=10)
    Button(root, text="Train Model", command=train_model, width=20).pack(pady=10)
    Button(root, text="Mark Attendance", command=mark_attendance, width=20).pack(pady=10)

    root.mainloop()
'''

# Run the GUI
if __name__ == "__main__":
    start_gui()
