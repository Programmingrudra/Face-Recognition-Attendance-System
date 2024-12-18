# Face-Recognition-Attendance-System
I have developed a Face recognition attendance system with the help of Python and Database with the help of which student can register their face and mark the attendance.
Purpose:
The Face Recognition Attendance System is designed to streamline and automate attendance tracking in various environments, such as educational institutions, workplaces, and events. By leveraging facial recognition technology, it replaces traditional, manual methods of marking attendance (e.g., roll calls or logbooks), reducing time, errors, and potential misuse.

How It Works:
Face Data Capture:
New users' facial data is captured and stored using a webcam. This data is processed, and multiple images are saved for training the recognition model.

Model Training:
A machine learning model (LBPH face recognizer) is trained on the captured data to create a unique facial profile for each user.

Recognition and Attendance:
When marking attendance, the system scans the face in real-time. It identifies the user, validates against the existing dataset, and marks them "Present" with a timestamp.

Database Integration:
User details (name, roll number, etc.) and attendance records are stored in a database or CSV files for reporting and analysis.

User-Friendly Interface:
The system provides a GUI for tasks like capturing new faces, training the model, and marking attendance, making it accessible even to non-technical users.
