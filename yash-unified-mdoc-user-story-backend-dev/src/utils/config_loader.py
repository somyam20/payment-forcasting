"""
Configuration Loader Utility
Handles loading and parsing YAML configuration files with environment variable substitution
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage YAML configuration files with environment variable substitution"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the config loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load YAML config file with environment variable substitution
        
        Args:
            config_file: Name of the config file (e.g., 'model_config.yaml')
            
        Returns:
            Dictionary containing configuration values
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Substitute environment variables in the format ${VAR_NAME}
            content = os.path.expandvars(content)
            
            config = yaml.safe_load(content)
            self._configs[config_file] = config
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            raise
    
    def get_config(self, config_file: str) -> Dict[str, Any]:
        """
        Get cached configuration or load if not cached
        
        Args:
            config_file: Name of the config file
            
        Returns:
            Dictionary containing configuration values
        """
        if config_file not in self._configs:
            return self.load_config(config_file)
        return self._configs[config_file]
    
    def get_value(self, config_file: str, path: str, default: Any = None) -> Any:
        """
        Get nested config value using dot notation
        
        Args:
            config_file: Name of the config file
            path: Dot-separated path to the value (e.g., 'models.default.temperature')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config(config_file)
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def reload_config(self, config_file: str) -> Dict[str, Any]:
        """
        Force reload a configuration file
        
        Args:
            config_file: Name of the config file
            
        Returns:
            Dictionary containing configuration values
        """
        if config_file in self._configs:
            del self._configs[config_file]
        return self.load_config(config_file)


# Global config loader instance
_config_loader = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """
    Get or create the global config loader instance
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def load_config(config_file: str, config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load a configuration file
    
    Args:
        config_file: Name of the config file
        config_dir: Directory containing configuration files
        
    Returns:
        Dictionary containing configuration values
    """
    loader = get_config_loader(config_dir)
    return loader.load_config(config_file)


def get_config_value(config_file: str, path: str, default: Any = None, config_dir: str = "config") -> Any:
    """
    Convenience function to get a configuration value
    
    Args:
        config_file: Name of the config file
        path: Dot-separated path to the value
        default: Default value if path not found
        config_dir: Directory containing configuration files
        
    Returns:
        Configuration value or default
    """
    loader = get_config_loader(config_dir)
    return loader.get_value(config_file, path, default)

