import os
from dotenv import load_dotenv
from roboflow import Roboflow

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from the environment
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE_NAME", "vehicle-damage-detection-93l5j") 
    project_name = os.getenv("ROBOFLOW_PROJECT_NAME", "front_kr")
    project_version = os.getenv("ROBOFLOW_PROJECT_VERSION", "1")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in .env file.")
        print("Please add it to your .env file like this:")
        print("ROBOFLOW_API_KEY=your_actual_api_key_here")
        return
        
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(int(project_version))
    
    # Download the dataset in coco format
    dataset = version.download("coco")
    
    print(f"Dataset successfully downloaded to: {dataset.location}")
    print("You can now move these files to your data/ directory if needed.")

if __name__ == "__main__":
    main()
