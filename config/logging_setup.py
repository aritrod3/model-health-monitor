"""Logging configuration setup."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: Optional[Path] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        config_path: Path to logging YAML config file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Override log file path
    """
    # Default config path
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "logging.yaml"
    
    # Check environment variable
    env_config_path = os.getenv("MODEL_MONITOR_LOGGING_CONFIG")
    if env_config_path:
        config_path = Path(env_config_path)
    
    # Load logging config if file exists
    if config_path.exists():
        with open(config_path, 'r') as f:
            logging_config = yaml.safe_load(f)
        
        # Override log level if provided
        if log_level:
            logging_config['root']['level'] = log_level.upper()
            for logger_config in logging_config.get('loggers', {}).values():
                logger_config['level'] = log_level.upper()
        
        # Override log file if provided
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Update file handlers
            for handler_config in logging_config.get('handlers', {}).values():
                if 'filename' in handler_config:
                    handler_config['filename'] = log_file
        
        # Ensure log directories exist
        for handler_config in logging_config.get('handlers', {}).values():
            if 'filename' in handler_config:
                log_dir = Path(handler_config['filename']).parent
                log_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply logging configuration
        logging.config.dictConfig(logging_config)
    else:
        # Fallback to basic config
        level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        handlers = [logging.StreamHandler()]
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=handlers
        )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
