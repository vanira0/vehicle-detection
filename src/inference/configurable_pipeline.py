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
import models.vehicle

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
            model_name = model_cfg.get("model_name")
            conf_thresh = model_cfg.get("confidence_threshold")
            
            try:
                model, wrapper, m_config = self._load_model(checkpoint_path, name, model_name)
                self.models.append({
                    "name": name,
                    "model": model,
                    "wrapper": wrapper,
                    "config": m_config,
                    "confidence_threshold": conf_thresh
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

    def _load_model(self, checkpoint_path: str, name: str, provided_model_name: Optional[str] = None):
        is_yolo = False
        if provided_model_name and "yolo" in provided_model_name.lower():
            is_yolo = True
        elif checkpoint_path.endswith('.pt') and not checkpoint_path.endswith('.pth'):
            is_yolo = True
            
        if is_yolo:
            from ultralytics import YOLO
            model_name = provided_model_name if provided_model_name else "yolo11_seg"
            model_wrapper = get_model(model_name)()
            yolo_model = YOLO(checkpoint_path)
            model_wrapper._yolo_model = yolo_model
            model = yolo_model.model.to(self.device)
            model.eval()
            self.logger.info(f"Loaded {name} YOLO model: {model_name}")
            m_config = Config.from_dict({"model": {"name": model_name}})
            return model, model_wrapper, m_config
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = Config.from_dict(checkpoint["config"])
        
        model_name = provided_model_name if provided_model_name else config.model.name
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
                threshold = m.get("confidence_threshold")
                if threshold is None:
                    threshold = self.confidence_threshold
                    
                if hasattr(wrapper, "_yolo_model"):
                    import cv2
                    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    preds = wrapper._yolo_model.predict(img_bgr, verbose=False, conf=threshold)
                    res = wrapper.post_process(preds, threshold)[0]
                else:
                    with torch.no_grad():
                        preds = model([det_image])
                    res = wrapper.post_process(preds, threshold)[0]
                    
                context[name] = res
                
                result[name] = {
                    "num_detections": len(res["labels"]),
                    "labels": res["labels"].tolist() if hasattr(res["labels"], "tolist") else res["labels"],
                    "scores": res["scores"].tolist() if hasattr(res["scores"], "tolist") else res["scores"],
                }
                
                if name == "vehicle" and hasattr(wrapper, "select_target_vehicle"):
                    target_idx = wrapper.select_target_vehicle(res, image_rgb.shape)
                    if target_idx is not None:
                        # Apply crop+mask to isolate the vehicle for downstream models
                        image_rgb = wrapper.extract_vehicle_roi(image_rgb, res, target_idx)
                        
                        # Recompute standardized images for downstream models
                        class_image = preprocess_image(image_rgb, target_size=224).to(self.device)
                        det_image = preprocess_for_detection(image_rgb, target_size=1024).to(self.device)
                    else:
                        # If no vehicle found, we might want to stop or continue with original
                        result["status"] = "no_vehicle_found"
                        break

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
