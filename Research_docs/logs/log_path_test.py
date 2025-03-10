import os
import logging

# Define log directory and file
log_dir = os.path.abspath(r"C:\Users\sathish\Downloads\FL_ModelForAV\logs")  # Ensure logs are directly in "logs/"
os.makedirs(log_dir, exist_ok=True)  # Create if not exists

log_file = os.path.join(log_dir, "yolo_conversion_json_to_yolov5_100k.log")

# Reset logging handlers (useful for Jupyter Notebook)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Test log entries
logging.info("This is a test log entry.")
logging.warning("This is a test warning message.")
logging.error("This is a test error message.")

# Check if the log file exists and read the last 5 lines
if os.path.exists(log_file):
    print(f"✅ Log file successfully created at: {log_file}")
    
    try:
        with open(log_file, 'r') as file:
            lines = file.readlines()
            print("\n📄 Last 5 log entries:")
            print("".join(lines[-5:]))  # Print last 5 log entries
    except Exception as e:
        print(f"⚠️ Error reading log file: {e}")
else:
    print("❌ Log file was not created. Check script permissions and path.")
