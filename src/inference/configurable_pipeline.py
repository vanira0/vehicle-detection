"""
Configurable Multi-Model Inference Pipeline
"""

import time
from typing import Any, Dict, List, Optional

import torch

from data.preprocessing import load_image, preprocess_for_detection, preprocess_image
from models.base import BaseClassifier, BaseDetector
from models.registry import get_model
from utils.config import Config
from utils.logger import setup_logger

# Import models to ensure they are registered in the registry
import models.gatekeeper
import models.angle
import models.damage
import models.parts

from .orchestrator import Orchestrator


class ConfigurablePipeline:
    def __init__(self, config_path: str):
        self.logger = setup_logger("configurable_pipeline")
        self.config = Config.from_file(config_path)
        
        device_str = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        self.models = []
        
        for model_cfg in self.config.get("models", []):
            name = model_cfg["name"]
            checkpoint_path = model_cfg["checkpoint"]
            
            try:
                model, wrapper, m_config = self._load_model(checkpoint_path, name)
                self.models.append({
                    "name": name,
                    "model": model,
                    "wrapper": wrapper,
                    "config": m_config
                })
            except FileNotFoundError:
                self.logger.warning(f"Checkpoint not found for model {name} at {checkpoint_path}. Skipping.")
            
        self.post_processors = self.config.get("post_processors", [])
        
        # Check if we need the damage_to_parts orchestrator
        if any(p.get("name") == "damage_to_parts_orchestrator" for p in self.post_processors):
            iou_thresh = self.config.get("iou_threshold", 0.3)
            
            damage_classes = None
            part_classes = None
            for m in self.models:
                if m["name"] == "damage" and m["config"].get("data.class_names") is not None:
                    damage_classes = m["config"].get("data.class_names")
                if m["name"] == "parts" and m["config"].get("data.class_names") is not None:
                    part_classes = m["config"].get("data.class_names")

            self.orchestrator = Orchestrator(
                iou_threshold=iou_thresh,
                damage_classes=damage_classes,
                part_classes=part_classes
            )
        else:
            self.orchestrator = None

        self.logger.info(f"ConfigurablePipeline ready with {len(self.models)} models on {self.device}.")

    def _load_model(self, checkpoint_path: str, name: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = Config.from_dict(checkpoint["config"])
        
        model_name = config.model.name
        model_wrapper = get_model(model_name)()
        
        model = model_wrapper.build(config.model).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        self.logger.info(f"Loaded {name} model: {model_name}")
        return model, model_wrapper, config

    def __call__(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        image_rgb = load_image(image_path)
        
        result = {
            "image_path": image_path,
            "status": "success",
            "inference_time_ms": 0,
            "findings": []
        }
        
        context = {
            "image_rgb": image_rgb
        }
        
        # Precompute standardized images
        # In a more advanced pipeline, preprocessing could be dictated by the wrapper
        class_image = preprocess_image(image_rgb, target_size=224).to(self.device)
        det_image = preprocess_for_detection(image_rgb, target_size=1024).to(self.device)
        
        for m in self.models:
            name = m["name"]
            model = m["model"]
            wrapper = m["wrapper"]
            
            if isinstance(wrapper, BaseClassifier):
                with torch.no_grad():
                    res = wrapper.predict(model, class_image)
                context[name] = res
                result[name] = res
                
                # Optional: specific logic for gatekeeper
                if name == "gatekeeper" and not res.get("is_damaged", True):
                    result["status"] = "no_damage"
                    break
                    
            elif isinstance(wrapper, BaseDetector):
                with torch.no_grad():
                    preds = model([det_image])
                
                res = wrapper.post_process(preds, self.confidence_threshold)[0]
                context[name] = res
                
                result[name] = {
                    "num_detections": len(res["labels"]),
                    "labels": res["labels"].tolist() if hasattr(res["labels"], "tolist") else res["labels"],
                    "scores": res["scores"].tolist() if hasattr(res["scores"], "tolist") else res["scores"],
                }

        # Post-processors execution
        if self.orchestrator and "damage" in context and "parts" in context:
            findings = self.orchestrator.map_damage_to_parts(context["damage"], context["parts"])
            result["findings"] = findings
            if findings:
                result["status"] = "damaged"

        result["inference_time_ms"] = (time.time() - start_time) * 1000
        
        # We attach the full context (including masks/boxes) to result (can be stripped later if not needed)
        # Note: Do not include full numpy arrays in normal JSON outputs, keep them in context for scripts to save
        result["_context"] = context 

        return result
