"""
model registry for sharing trained models across notebooks.
"""
import torch
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class ModelRegistry:
    """registry for managing trained models and their metadata."""
    
    def __init__(self, registry_dir: str = "experiments/models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """load existing metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, model_name: str, model: torch.nn.Module, 
                      config: Dict[str, Any], history: Dict[str, Any],
                      test_results: Optional[Dict[str, Any]] = None) -> str:
        """register a trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        
        model_path = self.registry_dir / f"{model_id}.pth"
        torch.save(model.state_dict(), model_path)
        
        self.metadata[model_id] = {
            "model_name": model_name,
            "model_path": str(model_path),
            "config": config,
            "history": history,
            "test_results": test_results,
            "created_at": timestamp,
            "parameters": sum(p.numel() for p in model.parameters())
        }
        
        self._save_metadata()
        return model_id
    
    def load_model(self, model_id: str, model_class, config: Dict[str, Any]) -> torch.nn.Module:
        """load a registered model."""
        if model_id not in self.metadata:
            raise ValueError(f"model {model_id} not found in registry")
        
        model_info = self.metadata[model_id]
        model_path = model_info["model_path"]
        
        model = model_class(**config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """get model metadata."""
        if model_id not in self.metadata:
            raise ValueError(f"model {model_id} not found in registry")
        return self.metadata[model_id]
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """list all registered models."""
        return self.metadata
    
    def get_latest_model(self, model_name: str) -> str:
        """get the latest model of a specific type."""
        matching_models = {k: v for k, v in self.metadata.items() 
                          if v["model_name"] == model_name}
        
        if not matching_models:
            raise ValueError(f"no models found for {model_name}")
        
        latest = max(matching_models.items(), 
                    key=lambda x: x[1]["created_at"])
        return latest[0]
    
    def delete_model(self, model_id: str):
        """delete a model from registry."""
        if model_id not in self.metadata:
            raise ValueError(f"model {model_id} not found in registry")
        
        model_info = self.metadata[model_id]
        model_path = Path(model_info["model_path"])
        
        if model_path.exists():
            model_path.unlink()
        
        del self.metadata[model_id]
        self._save_metadata()

# global registry instance
registry = ModelRegistry()



