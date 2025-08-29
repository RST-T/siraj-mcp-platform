"""
AI Model Interface for SIRAJ v6.1
Provides a comprehensive interface for users to register and use their own AI models
with web search capabilities, semantic analysis, and cultural validation support.
Implements the user-driven validation model described in the Implementation Guide.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
import asyncio
import aiohttp
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models supported by the interface"""
    OPENAI_COMPATIBLE = "openai_compatible"
    HUGGINGFACE = "huggingface"
    CUSTOM_ENDPOINT = "custom_endpoint"
    LOCAL_MODEL = "local_model"
    MOCK_MODEL = "mock_model"

class ModelCapability(Enum):
    """Capabilities that AI models can provide"""
    TEXT_ANALYSIS = "text_analysis"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    WEB_SEARCH = "web_search"
    CULTURAL_VALIDATION = "cultural_validation"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"
    SYNTACTIC_PARSING = "syntactic_parsing"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"

class ValidationLevel(Enum):
    """Validation levels for cultural sensitivity"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    COMMUNITY_CONSENSUS = "community_consensus"
    CULTURAL_AUTHORITY = "cultural_authority"

@dataclass
class ModelMetadata:
    """Metadata for AI model registration"""
    model_id: str
    user_id: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    description: str = ""
    cultural_specializations: List[str] = field(default_factory=list)
    language_support: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    registration_timestamp: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per minute
    cost_per_call: float = 0.0005  # Default cost as per Implementation Guide
    community_validation_required: bool = False

class AIModelInput(BaseModel):
    """Comprehensive input schema for user-provided AI models"""
    text: str
    context: Optional[Dict[str, Any]] = None
    capability: ModelCapability
    cultural_context: Optional[str] = None
    language: str = "auto_detect"
    validation_level: ValidationLevel = ValidationLevel.BASIC
    additional_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text input cannot be empty')
        return v

class AIModelOutput(BaseModel):
    """Comprehensive output schema for user-provided AI models"""
    result: Any
    model_id: str
    capability_used: ModelCapability
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    cultural_sensitivity_score: Optional[float] = None
    validation_status: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    cost_incurred: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

@dataclass
class MockModelResponse:
    """Response from mock models for testing"""
    result: Any
    confidence: float
    processing_time: float
    cultural_sensitivity_score: float = 0.85
    metadata: Dict[str, Any] = field(default_factory=dict)

class AIModelInterface:
    """Comprehensive interface for managing user-provided AI models
    
    Supports the user-driven validation model where users provide their own AI models
    with web search capabilities for community validation tasks.
    """

    def __init__(self, config_settings=None):
        self.models: Dict[str, ModelMetadata] = {}
        self.model_sessions: Dict[str, aiohttp.ClientSession] = {}
        self.usage_tracker: Dict[str, List[datetime]] = {}
        self.config = config_settings or {}
        self.free_credit_limit = 5.0  # $5 free credits as per Implementation Guide
        self.user_credits: Dict[str, float] = {}  # Track user credits
        self.mock_models = self._initialize_mock_models()

    def _initialize_mock_models(self) -> Dict[str, ModelMetadata]:
        """Initialize mock AI models for testing and demonstration"""
        mock_models = {}
        
        # Mock OpenAI-compatible model
        mock_models["mock_openai_gpt4"] = ModelMetadata(
            model_id="mock_openai_gpt4",
            user_id="system",
            model_type=ModelType.MOCK_MODEL,
            capabilities=[
                ModelCapability.TEXT_ANALYSIS,
                ModelCapability.SEMANTIC_SIMILARITY,
                ModelCapability.WEB_SEARCH,
                ModelCapability.CULTURAL_VALIDATION
            ],
            model_name="Mock GPT-4 Compatible",
            description="Mock model simulating OpenAI GPT-4 capabilities",
            cultural_specializations=["islamic_studies", "comparative_linguistics"],
            language_support=["ar", "en", "he", "fa", "ur"],
            performance_metrics={"accuracy": 0.89, "cultural_sensitivity": 0.92},
            cost_per_call=0.0001  # Lower cost for mock model
        )
        
        # Mock HuggingFace model
        mock_models["mock_huggingface_bert"] = ModelMetadata(
            model_id="mock_huggingface_bert",
            user_id="system",
            model_type=ModelType.MOCK_MODEL,
            capabilities=[
                ModelCapability.TEXT_ANALYSIS,
                ModelCapability.MORPHOLOGICAL_ANALYSIS,
                ModelCapability.NAMED_ENTITY_RECOGNITION
            ],
            model_name="Mock BERT Multilingual",
            description="Mock model simulating BERT multilingual capabilities",
            cultural_specializations=["semitic_languages"],
            language_support=["ar", "en", "he"],
            performance_metrics={"accuracy": 0.85, "cultural_sensitivity": 0.78},
            cost_per_call=0.0002
        )
        
        # Mock cultural validation model
        mock_models["mock_cultural_validator"] = ModelMetadata(
            model_id="mock_cultural_validator",
            user_id="system",
            model_type=ModelType.MOCK_MODEL,
            capabilities=[ModelCapability.CULTURAL_VALIDATION],
            model_name="Mock Cultural Validation Model",
            description="Specialized mock model for cultural sensitivity validation",
            cultural_specializations=["islamic_studies", "indigenous_knowledge", "traditional_practices"],
            language_support=["ar", "en", "he", "fa", "ur", "sw"],
            performance_metrics={"accuracy": 0.94, "cultural_sensitivity": 0.98},
            cost_per_call=0.0003,
            community_validation_required=True
        )
        
        return mock_models
    
    async def register_model(self, 
                           model_id: str,
                           user_id: str,
                           model_type: ModelType,
                           capabilities: List[ModelCapability],
                           endpoint_url: Optional[str] = None,
                           api_key: Optional[str] = None,
                           model_name: Optional[str] = None,
                           description: str = "",
                           cultural_specializations: List[str] = None,
                           language_support: List[str] = None,
                           rate_limit: Optional[int] = None,
                           cost_per_call: float = 0.0005) -> bool:
        """Register a comprehensive AI model with full metadata"""
        try:
            # Validate model registration
            if model_id in self.models:
                logger.warning(f"Model {model_id} already registered, updating...")
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                user_id=user_id,
                model_type=model_type,
                capabilities=capabilities,
                endpoint_url=endpoint_url,
                api_key=api_key,
                model_name=model_name or model_id,
                description=description,
                cultural_specializations=cultural_specializations or [],
                language_support=language_support or ["en"],
                rate_limit=rate_limit,
                cost_per_call=cost_per_call
            )
            
            # Test model connectivity for non-mock models
            if model_type != ModelType.MOCK_MODEL and endpoint_url:
                connectivity_test = await self._test_model_connectivity(metadata)
                if not connectivity_test:
                    logger.error(f"Model {model_id} failed connectivity test")
                    return False
            
            # Register model
            self.models[model_id] = metadata
            
            # Initialize user credits if new user
            if user_id not in self.user_credits:
                self.user_credits[user_id] = self.free_credit_limit
            
            logger.info(f"AI model registered successfully: {model_id} by user {user_id}")
            logger.info(f"Model capabilities: {[cap.value for cap in capabilities]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering AI model {model_id}: {str(e)}")
            return False

    async def _test_model_connectivity(self, metadata: ModelMetadata) -> bool:
        """Test connectivity and basic functionality of a registered model"""
        try:
            if not metadata.endpoint_url:
                return True  # Skip test for models without endpoints
                
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            
            # Prepare test payload
            test_payload = {
                "text": "Test connectivity",
                "capability": "text_analysis"
            }
            
            headers = {"Content-Type": "application/json"}
            if metadata.api_key:
                headers["Authorization"] = f"Bearer {metadata.api_key}"
            
            async with session.post(
                metadata.endpoint_url,
                json=test_payload,
                headers=headers
            ) as response:
                success = response.status == 200
                await session.close()
                return success
                
        except Exception as e:
            logger.warning(f"Model connectivity test failed for {metadata.model_id}: {str(e)}")
            return False
    
    def list_models(self, user_id: Optional[str] = None, 
                   capability_filter: Optional[ModelCapability] = None,
                   cultural_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List registered AI models with optional filtering"""
        filtered_models = {}
        
        for model_id, metadata in self.models.items():
            # Apply user filter
            if user_id and metadata.user_id != user_id and metadata.user_id != "system":
                continue
                
            # Apply capability filter
            if capability_filter and capability_filter not in metadata.capabilities:
                continue
                
            # Apply cultural specialization filter
            if cultural_filter and cultural_filter not in metadata.cultural_specializations:
                continue
            
            # Prepare safe metadata for return (exclude sensitive info)
            safe_metadata = {
                "model_id": metadata.model_id,
                "model_type": metadata.model_type.value,
                "capabilities": [cap.value for cap in metadata.capabilities],
                "model_name": metadata.model_name,
                "description": metadata.description,
                "cultural_specializations": metadata.cultural_specializations,
                "language_support": metadata.language_support,
                "performance_metrics": metadata.performance_metrics,
                "cost_per_call": metadata.cost_per_call,
                "registration_timestamp": metadata.registration_timestamp.isoformat(),
                "usage_count": metadata.usage_count,
                "community_validation_required": metadata.community_validation_required
            }
            
            filtered_models[model_id] = safe_metadata
            
        # Include mock models
        for mock_id, mock_metadata in self.mock_models.items():
            if mock_id not in filtered_models:  # Don't duplicate if already added
                # Apply same filters to mock models
                if capability_filter and capability_filter not in mock_metadata.capabilities:
                    continue
                if cultural_filter and cultural_filter not in mock_metadata.cultural_specializations:
                    continue
                    
                filtered_models[mock_id] = {
                    "model_id": mock_metadata.model_id,
                    "model_type": mock_metadata.model_type.value,
                    "capabilities": [cap.value for cap in mock_metadata.capabilities],
                    "model_name": mock_metadata.model_name,
                    "description": mock_metadata.description,
                    "cultural_specializations": mock_metadata.cultural_specializations,
                    "language_support": mock_metadata.language_support,
                    "performance_metrics": mock_metadata.performance_metrics,
                    "cost_per_call": mock_metadata.cost_per_call,
                    "registration_timestamp": mock_metadata.registration_timestamp.isoformat(),
                    "usage_count": mock_metadata.usage_count,
                    "community_validation_required": mock_metadata.community_validation_required,
                    "is_mock": True
                }
        
        return filtered_models

    async def call_model(self, 
                       model_id: str, 
                       data: AIModelInput, 
                       user_id: str) -> Optional[AIModelOutput]:
        """Call a registered AI model with comprehensive handling"""
        start_time = datetime.now()
        
        try:
            # Check if model exists
            if model_id not in self.models and model_id not in self.mock_models:
                logger.error(f"AI model not found: {model_id}")
                return None
            
            # Get model metadata
            if model_id in self.mock_models:
                metadata = self.mock_models[model_id]
            else:
                metadata = self.models[model_id]
            
            # Check if user has permission to use this model
            if metadata.user_id != "system" and metadata.user_id != user_id:
                logger.error(f"User {user_id} does not have permission to use model {model_id}")
                return None
            
            # Check if model supports requested capability
            if data.capability not in metadata.capabilities:
                logger.error(f"Model {model_id} does not support capability {data.capability.value}")
                return None
            
            # Check user credits
            user_credits = self.user_credits.get(user_id, 0.0)
            if user_credits < metadata.cost_per_call:
                logger.error(f"Insufficient credits for user {user_id}. Required: {metadata.cost_per_call}, Available: {user_credits}")
                return None
            
            # Check rate limiting
            if not self._check_rate_limit(user_id, metadata):
                logger.warning(f"Rate limit exceeded for user {user_id} on model {model_id}")
                return None
            
            # Process the model call
            if metadata.model_type == ModelType.MOCK_MODEL:
                result = await self._call_mock_model(metadata, data)
            else:
                result = await self._call_real_model(metadata, data)
            
            if result is None:
                return None
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Deduct credits
            self.user_credits[user_id] -= metadata.cost_per_call
            
            # Update usage statistics
            metadata.usage_count += 1
            metadata.last_used = datetime.now()
            
            # Create comprehensive output
            output = AIModelOutput(
                result=result["result"],
                model_id=model_id,
                capability_used=data.capability,
                confidence=result.get("confidence", 0.85),
                processing_time=processing_time,
                cultural_sensitivity_score=result.get("cultural_sensitivity_score"),
                validation_status=result.get("validation_status"),
                metadata={
                    "model_name": metadata.model_name,
                    "model_type": metadata.model_type.value,
                    "user_remaining_credits": self.user_credits[user_id],
                    "cultural_context": data.cultural_context,
                    "language": data.language,
                    "validation_level": data.validation_level.value
                },
                cost_incurred=metadata.cost_per_call
            )
            
            logger.info(f"Successfully called model {model_id} for user {user_id}")
            return output
            
        except Exception as e:
            logger.error(f"Error calling AI model {model_id}: {str(e)}")
            return None
    
    async def _call_mock_model(self, metadata: ModelMetadata, data: AIModelInput) -> Optional[Dict[str, Any]]:
        """Handle calls to mock AI models for testing"""
        try:
            # Simulate processing delay
            await asyncio.sleep(0.1 + np.random.exponential(0.05))
            
            # Generate mock responses based on capability
            if data.capability == ModelCapability.TEXT_ANALYSIS:
                result = {
                    "analysis": {
                        "sentiment": "neutral",
                        "topics": ["linguistics", "computation"],
                        "complexity_score": np.random.uniform(0.3, 0.9),
                        "readability": np.random.uniform(0.5, 0.95)
                    },
                    "tokens": len(data.text.split()),
                    "language_detected": data.language
                }
            
            elif data.capability == ModelCapability.SEMANTIC_SIMILARITY:
                # Mock semantic similarity calculation
                similarity_score = np.random.uniform(0.1, 0.95)
                result = {
                    "similarity_score": similarity_score,
                    "semantic_features": ["conceptual", "contextual", "structural"],
                    "confidence": min(0.95, similarity_score + 0.1)
                }
            
            elif data.capability == ModelCapability.WEB_SEARCH:
                # Mock web search results
                result = {
                    "search_results": [
                        {
                            "title": "Mock Search Result 1",
                            "url": "https://example.com/result1",
                            "snippet": "Relevant information about the query",
                            "relevance_score": 0.89
                        },
                        {
                            "title": "Mock Search Result 2",
                            "url": "https://example.com/result2", 
                            "snippet": "Additional context and information",
                            "relevance_score": 0.76
                        }
                    ],
                    "total_results": 2,
                    "query_processed": data.text[:100]
                }
            
            elif data.capability == ModelCapability.CULTURAL_VALIDATION:
                # Mock cultural validation analysis
                cultural_score = np.random.uniform(0.7, 0.98)
                result = {
                    "cultural_appropriateness": cultural_score,
                    "sensitivity_flags": [],
                    "recommendations": ["Consider cultural context", "Verify with community authorities"],
                    "validation_level": data.validation_level.value,
                    "cultural_context_analysis": {
                        "context": data.cultural_context or "general",
                        "relevant_traditions": ["scholarly", "linguistic"],
                        "potential_concerns": []
                    }
                }
            
            elif data.capability == ModelCapability.MORPHOLOGICAL_ANALYSIS:
                # Mock morphological analysis
                words = data.text.split()
                result = {
                    "morphological_breakdown": [
                        {
                            "word": word,
                            "root": word[:3] if len(word) > 3 else word,
                            "pattern": "mock_pattern",
                            "morphemes": [word[:len(word)//2], word[len(word)//2:]],
                            "pos": "NOUN" if i % 2 == 0 else "VERB"
                        } for i, word in enumerate(words[:5])  # Limit to first 5 words
                    ],
                    "language": data.language,
                    "confidence": np.random.uniform(0.75, 0.92)
                }
            
            else:
                # Default mock response
                result = {
                    "response": f"Mock {data.capability.value} analysis completed",
                    "input_processed": data.text[:100],
                    "capability": data.capability.value
                }
            
            return {
                "result": result,
                "confidence": np.random.uniform(0.8, 0.95),
                "cultural_sensitivity_score": np.random.uniform(0.85, 0.98),
                "validation_status": "mock_validated"
            }
            
        except Exception as e:
            logger.error(f"Error in mock model call: {str(e)}")
            return None
    
    async def _call_real_model(self, metadata: ModelMetadata, data: AIModelInput) -> Optional[Dict[str, Any]]:
        """Handle calls to real AI models"""
        try:
            if not metadata.endpoint_url:
                logger.error(f"No endpoint URL configured for model {metadata.model_id}")
                return None
            
            # Prepare request payload
            payload = {
                "text": data.text,
                "capability": data.capability.value,
                "context": data.context or {},
                "cultural_context": data.cultural_context,
                "language": data.language,
                "validation_level": data.validation_level.value,
                "additional_parameters": data.additional_parameters
            }
            
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if metadata.api_key:
                headers["Authorization"] = f"Bearer {metadata.api_key}"
            
            # Get or create session for this model
            if metadata.model_id not in self.model_sessions:
                self.model_sessions[metadata.model_id] = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
            
            session = self.model_sessions[metadata.model_id]
            
            # Make the API call
            async with session.post(
                metadata.endpoint_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return {
                        "result": response_data.get("result", response_data),
                        "confidence": response_data.get("confidence", 0.85),
                        "cultural_sensitivity_score": response_data.get("cultural_sensitivity_score"),
                        "validation_status": response_data.get("validation_status", "api_processed")
                    }
                else:
                    logger.error(f"Model API call failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling real AI model {metadata.model_id}: {str(e)}")
            return None
    
    def _check_rate_limit(self, user_id: str, metadata: ModelMetadata) -> bool:
        """Check if user has exceeded rate limit for this model"""
        if not metadata.rate_limit:
            return True  # No rate limit configured
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Initialize usage tracking for user if needed
        user_key = f"{user_id}_{metadata.model_id}"
        if user_key not in self.usage_tracker:
            self.usage_tracker[user_key] = []
        
        # Clean old entries
        self.usage_tracker[user_key] = [
            timestamp for timestamp in self.usage_tracker[user_key]
            if timestamp > minute_ago
        ]
        
        # Check rate limit
        if len(self.usage_tracker[user_key]) >= metadata.rate_limit:
            return False
        
        # Record this usage
        self.usage_tracker[user_key].append(now)
        return True
    
    def get_user_credits(self, user_id: str) -> float:
        """Get remaining credits for a user"""
        return self.user_credits.get(user_id, 0.0)
    
    def add_user_credits(self, user_id: str, amount: float) -> bool:
        """Add credits to a user's account"""
        try:
            if user_id not in self.user_credits:
                self.user_credits[user_id] = self.free_credit_limit
            
            self.user_credits[user_id] += amount
            logger.info(f"Added ${amount} credits to user {user_id}. New balance: ${self.user_credits[user_id]}")
            return True
        except Exception as e:
            logger.error(f"Error adding credits for user {user_id}: {str(e)}")
            return False
    
    def get_model_usage_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a specific model"""
        if model_id in self.models:
            metadata = self.models[model_id]
        elif model_id in self.mock_models:
            metadata = self.mock_models[model_id]
        else:
            return None
        
        return {
            "model_id": model_id,
            "total_usage_count": metadata.usage_count,
            "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
            "registration_date": metadata.registration_timestamp.isoformat(),
            "cost_per_call": metadata.cost_per_call,
            "total_revenue": metadata.usage_count * metadata.cost_per_call
        }
    
    async def cleanup(self):
        """Clean up resources and close sessions"""
        try:
            for session in self.model_sessions.values():
                await session.close()
            self.model_sessions.clear()
            logger.info("AI Model Interface cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")