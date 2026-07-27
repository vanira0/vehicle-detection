import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from inference.configurable_pipeline import ConfigurablePipeline

pipeline = ConfigurablePipeline(config_path="configs/pipeline/test_all.yaml")
for m in pipeline.models:
    name = m["name"]
    print(f"Model: {name}")
    class_names = None
    if hasattr(m.get("wrapper"), "_yolo_model"):
        class_names = m["wrapper"]._yolo_model.names
        print("  found _yolo_model.names:", class_names)
    elif hasattr(m.get("wrapper"), "names"):
        class_names = m["wrapper"].names
        print("  found .names:", class_names)
    elif hasattr(m.get("wrapper"), "model") and hasattr(m["wrapper"].model, "names"):
        class_names = m["wrapper"].model.names
        print("  found .model.names:", class_names)
    else:
        print("  could not find names in wrapper")
        print("  wrapper attributes:", dir(m.get("wrapper")))
