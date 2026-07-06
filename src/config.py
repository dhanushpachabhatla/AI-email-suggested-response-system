"""
Configuration management for AI Email Response System.

This module loads and validates configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass


class Config:
    """
    Configuration loader and validator for the AI Email Response System.
    
    Loads configuration from:
    1. YAML config file (config.yaml by default)
    2. Environment variables (.env file)
    3. Provides validation and defaults
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            ConfigurationError: If configuration is invalid or required fields missing
        """
        self.config_path = Path(config_path)
        
        # Load environment variables
        load_dotenv()
        
        # Load and validate configuration
        self.config = self._load_config()
        self._validate_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
            
        Raises:
            ConfigurationError: If config file cannot be loaded
        """
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ConfigurationError("Configuration file is empty")
            
            return config
        
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {e}")
    
    def _validate_config(self) -> None:
        """
        Validate required configuration fields and values.
        
        Raises:
            ConfigurationError: If validation fails
        """
        # Required top-level sections
        required_sections = ['dataset', 'embeddings', 'vector_store', 'llm', 
                            'generation', 'evaluation', 'output', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(
                    f"Missing required configuration section: {section}"
                )
        
        # Validate dataset configuration
        dataset = self.config['dataset']
        if 'path' not in dataset:
            raise ConfigurationError("dataset.path is required")
        if 'min_pairs' not in dataset or dataset['min_pairs'] < 1:
            raise ConfigurationError("dataset.min_pairs must be >= 1")
        
        # Validate embeddings configuration
        embeddings = self.config['embeddings']
        if 'model' not in embeddings:
            raise ConfigurationError("embeddings.model is required")
        
        # Validate LLM configuration
        llm = self.config['llm']
        if 'primary' not in llm:
            raise ConfigurationError("llm.primary is required")
        
        if llm['primary'] == 'lm_studio':
            if 'lm_studio_url' not in llm:
                raise ConfigurationError(
                    "llm.lm_studio_url is required when primary is lm_studio"
                )
        
        # Validate fallback configuration
        if llm.get('fallback_provider'):
            if llm['fallback_provider'] not in ['openai', 'anthropic']:
                raise ConfigurationError(
                    "llm.fallback_provider must be 'openai' or 'anthropic'"
                )
        
        # Validate generation configuration
        generation = self.config['generation']
        if not 0 <= generation.get('temperature', 0.7) <= 2:
            raise ConfigurationError("generation.temperature must be between 0 and 2")
        if generation.get('max_tokens', 500) < 1:
            raise ConfigurationError("generation.max_tokens must be >= 1")
        if generation.get('top_k_examples', 3) < 1:
            raise ConfigurationError("generation.top_k_examples must be >= 1")
        
        # Validate evaluation configuration
        evaluation = self.config['evaluation']
        if not evaluation.get('dimensions'):
            raise ConfigurationError("evaluation.dimensions cannot be empty")
        
        # Validate thresholds
        thresholds = evaluation.get('thresholds', {})
        for quality_level in ['high_quality', 'acceptable']:
            if quality_level not in thresholds:
                raise ConfigurationError(
                    f"evaluation.thresholds.{quality_level} is required"
                )
        
        # Validate output configuration
        output = self.config['output']
        if output.get('format') not in ['json', 'yaml', 'csv']:
            raise ConfigurationError("output.format must be 'json', 'yaml', or 'csv'")
        
        # Validate logging configuration
        logging = self.config['logging']
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if logging.get('level', 'INFO') not in valid_levels:
            raise ConfigurationError(
                f"logging.level must be one of {valid_levels}"
            )
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        Environment variables take precedence over config file values.
        """
        # LM Studio URL override
        if os.getenv('LM_STUDIO_URL'):
            self.config['llm']['lm_studio_url'] = os.getenv('LM_STUDIO_URL')
        
        # Fallback provider override
        if os.getenv('FALLBACK_PROVIDER'):
            self.config['llm']['fallback_provider'] = os.getenv('FALLBACK_PROVIDER')
        
        # API keys from environment
        if os.getenv('OPENAI_API_KEY'):
            self.config['llm']['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('ANTHROPIC_API_KEY'):
            self.config['llm']['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        # Config path override
        if os.getenv('CONFIG_PATH') and not hasattr(self, '_config_path_from_env'):
            # Avoid infinite recursion
            self._config_path_from_env = True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "llm.temperature")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("llm.lm_studio_url")
            "http://127.0.0.1:1234/v1"
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "llm", "dataset")
            
        Returns:
            Configuration section dictionary
            
        Raises:
            ConfigurationError: If section doesn't exist
        """
        if section not in self.config:
            raise ConfigurationError(f"Configuration section not found: {section}")
        
        return self.config[section]
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path='{self.config_path}')"
    
    def __str__(self) -> str:
        """Human-readable configuration summary."""
        return (
            f"AI Email Response System Configuration\n"
            f"  Config File: {self.config_path}\n"
            f"  Dataset: {self.get('dataset.path')}\n"
            f"  Embedding Model: {self.get('embeddings.model')}\n"
            f"  Primary LLM: {self.get('llm.primary')}\n"
            f"  LM Studio URL: {self.get('llm.lm_studio_url')}\n"
            f"  Fallback Provider: {self.get('llm.fallback_provider', 'None')}\n"
        )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file (uses environment variable or default)
        
    Returns:
        Loaded and validated Config object
    """
    if config_path is None:
        config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    
    return Config(config_path)
