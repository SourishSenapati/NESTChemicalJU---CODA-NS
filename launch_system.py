import os
import subprocess
import time
import sys

def main():
    print("Launching CODA-350M System...")
    
    # Check if files exist
    if not os.path.exists("app.py"):
        print("Error: app.py not found.")
        return

    # 1. Backend
    print("Starting Backend (Logic Engine)...")
    # Use 'start' on windows to open new windows, or Popen for background
    if sys.platform == "win32":
        os.system("start cmd /k python app.py")
    else:
        subprocess.Popen(["python", "app.py"])
    
    # 2. Frontend
    print("Starting Frontend Dashboard...")
    if sys.platform == "win32":
        os.system("start cmd /k python serve_frontend.py")
        os.system("start http://localhost:3000/index.html")
    else:
        subprocess.Popen(["python", "serve_frontend.py"])
    
    print("\n[SUCCESS] System Launching...")
    print("Monitor the popup windows for status.")

if __name__ == "__main__":
    main()
