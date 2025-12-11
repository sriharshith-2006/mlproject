import logging
import os
from datetime import datetime
LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)
if __name__=="__main__":
    logging.info("Logging has started")
    logging.error("This is an error message")
    logging.warning("This is a warning message")
    logging.info("Logging has ended")
    logging.info("This is an info message")
    logging.debug("This is a debug message")
    logging.critical("This is a critical message")
    logging.info("Logging test completed")
    print(f"Log file created at: {LOG_FILE_PATH}")
    
