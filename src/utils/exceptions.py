"""
Custom exceptions for the SIRAJ MCP Server.
"""

from datetime import datetime
from typing import Dict, Any, Optional

class SirajProcessingError(Exception):
    """Base exception for errors in the SIRAJ engine."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class ValidationError(SirajProcessingError):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field_name: str = None, expected_type: str = None, 
                 received_value: Any = None, validation_rules: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", {
            "field_name": field_name,
            "expected_type": expected_type,
            "received_value": str(received_value) if received_value is not None else None,
            "validation_rules": validation_rules or {}
        })

class CommunityValidationRequired(SirajProcessingError):
    """Exception raised when community validation is required."""
    def __init__(self, message, validation_details):
        super().__init__(message)
        self.validation_details = validation_details

class ArchitectureError(SirajProcessingError):
    """Exception raised for errors in the adaptive semantic architecture."""
    
    def __init__(self, message: str, architecture_component: str = None, 
                 node_id: str = None, operation: str = None, state_info: Dict[str, Any] = None):
        super().__init__(message, "ARCHITECTURE_ERROR", {
            "architecture_component": architecture_component,
            "node_id": node_id,
            "operation": operation,
            "state_info": state_info or {}
        })
