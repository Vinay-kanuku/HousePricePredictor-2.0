import yaml 
from pathlib import Path
import os 
import logging 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class ConfigManager:
    def __init__(self,):
        #This is the root directory of the project .. 
        self.base_dir = Path(__file__).resolve().parent.parent 
        try:
            file_path  = os.path.join(self.base_dir, "config", "config.yaml")
            with open(file_path, "r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError as e:
            logging.error(f"No such file {file_path}")
        except Exception as e:
            logging.error(f"Error while loading the file {file_path}")


    def get_path(self, file):
        file_path = self.config['paths'][file]
        file_path = os.path.join(self.base_dir, file_path)
        return file_path 
         

        
   


if __name__ == "__main__":
    ob = ConfigManager()
    print(ob.get_path('interim_data'))