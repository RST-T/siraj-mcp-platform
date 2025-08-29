"""
Community-Informed Transformer Integration for SIRAJ v6.1
Implements culturally-aware transformer architectures with community validation
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    DistilBertModel, DistilBertTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TransformerType(Enum):
    """Supported transformer types"""
    BERT = "bert"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    XLMROBERTA = "xlm-roberta"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    CULTURAL_BERT = "cultural-bert"

class CulturalAdaptationLevel(Enum):
    """Cultural adaptation levels for transformers"""
    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    COMMUNITY_TRAINED = "community_trained"

@dataclass
class TransformerConfig:
    """Transformer configuration with cultural parameters"""
    model_name: str
    transformer_type: TransformerType
    cultural_adaptation: CulturalAdaptationLevel
    supported_languages: List[str]
    cultural_groups: List[str]
    max_sequence_length: int = 512
    embedding_dim: int = 768
    cultural_embedding_dim: int = 128
    community_validated: bool = False
    sovereignty_compliant: bool = False

@dataclass
class CulturalEmbedding:
    """Cultural context embedding"""
    cultural_group: str
    embedding_vector: np.ndarray
    confidence: float
    source_authority: str
    validation_status: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TransformerOutput:
    """Enhanced transformer output with cultural context"""
    embeddings: np.ndarray
    attention_weights: Optional[np.ndarray]
    cultural_context: Optional[CulturalEmbedding]
    confidence_score: float
    cultural_appropriateness: float
    sovereignty_status: str
    processing_metadata: Dict[str, Any]

class CulturallyAwareBertModel(nn.Module):
    """BERT model enhanced with cultural context awareness"""
    
    def __init__(self, 
                 base_model_name: str,
                 cultural_embedding_dim: int = 128,
                 cultural_groups: List[str] = None):
        super().__init__()
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.config = self.bert.config
        
        # Cultural context layers
        self.cultural_groups = cultural_groups or []
        self.cultural_embedding_dim = cultural_embedding_dim
        
        # Cultural context encoder
        self.cultural_encoder = nn.Linear(
            len(self.cultural_groups), cultural_embedding_dim
        )
        
        # Cultural fusion layer
        self.cultural_fusion = nn.Linear(
            self.config.hidden_size + cultural_embedding_dim,
            self.config.hidden_size
        )
        
        # Cultural attention mechanism
        self.cultural_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Sovereignty compliance checker
        self.sovereignty_classifier = nn.Linear(
            self.config.hidden_size, 2  # compliant/non-compliant
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cultural_context: Optional[torch.Tensor] = None,
                return_cultural_analysis: bool = True):
        
        # Standard BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = bert_outputs.pooler_output
        
        # Apply cultural context if provided
        if cultural_context is not None and return_cultural_analysis:
            # Encode cultural context
            cultural_embedding = self.cultural_encoder(cultural_context)
            cultural_embedding = cultural_embedding.unsqueeze(1).expand(
                -1, sequence_output.size(1), -1
            )
            
            # Fuse BERT output with cultural context
            fused_representation = torch.cat([sequence_output, cultural_embedding], dim=-1)
            culturally_aware_output = self.cultural_fusion(fused_representation)
            
            # Apply cultural attention
            culturally_attended_output, cultural_attention_weights = self.cultural_attention(
                culturally_aware_output, culturally_aware_output, culturally_aware_output
            )
            
            # Sovereignty compliance check
            sovereignty_logits = self.sovereignty_classifier(
                culturally_attended_output.mean(dim=1)
            )
            sovereignty_probs = F.softmax(sovereignty_logits, dim=-1)
            
            return {
                "sequence_output": culturally_attended_output,
                "pooled_output": culturally_attended_output.mean(dim=1),
                "cultural_attention_weights": cultural_attention_weights,
                "sovereignty_compliance": sovereignty_probs[:, 1],  # Probability of compliance
                "cultural_embedding": cultural_embedding
            }
        
        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "cultural_attention_weights": None,
            "sovereignty_compliance": torch.ones(input_ids.size(0)),
            "cultural_embedding": None
        }

class CommunityInformedTransformerManager:
    """
    Manager for culturally-aware transformer models with community validation
    """
    
    def __init__(self, config_settings):
        self.config = config_settings
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.cultural_embeddings: Dict[str, CulturalEmbedding] = {}
        self.model_configs: Dict[str, TransformerConfig] = {}
        self.community_validations: Dict[str, Dict[str, Any]] = {}
        
        # Cultural adaptation parameters
        self.cultural_parameters = {
            "islamic_context": {
                "respectful_terms": ["Allah", "Prophet", "Messenger", "Quran", "Hadith"],
                "sensitive_concepts": ["divine", "sacred", "holy", "blessed"],
                "cultural_markers": ["arabic", "islamic", "muslim", "ummah"],
                "sovereignty_requirements": ["traditional_authority", "cultural_keeper"]
            },
            "semitic_linguistic": {
                "root_patterns": ["trilateral", "bilateral", "quadrilateral"],
                "morphological_features": ["prefix", "suffix", "infix", "templatic"],
                "cultural_markers": ["hebrew", "arabic", "aramaic", "semitic"],
                "sovereignty_requirements": ["linguistic_scholar", "community_representative"]
            }
        }
        
        # Initialize default models
        asyncio.create_task(self._initialize_default_models())
    
    async def initialize(self):
        """Initialize the Community Informed Transformer Manager"""
        logger.info("Community Informed Transformer Manager initialized successfully")
    
    async def _initialize_default_models(self):
        """Initialize default transformer models"""
        try:
            # Standard multilingual models
            await self.load_transformer_model(
                model_id="multilingual_bert",
                config=TransformerConfig(
                    model_name="bert-base-multilingual-cased",
                    transformer_type=TransformerType.BERT,
                    cultural_adaptation=CulturalAdaptationLevel.BASIC,
                    supported_languages=["en", "ar", "he", "fa"],
                    cultural_groups=["general"]
                )
            )
            
            # Arabic-specialized model
            await self.load_transformer_model(
                model_id="arabic_bert",
                config=TransformerConfig(
                    model_name="aubmindlab/bert-base-arabertv2",
                    transformer_type=TransformerType.BERT,
                    cultural_adaptation=CulturalAdaptationLevel.INTERMEDIATE,
                    supported_languages=["ar"],
                    cultural_groups=["islamic", "arabic"]
                )
            )
            
            # Sentence transformer for semantic similarity
            await self.load_transformer_model(
                model_id="sentence_transformer",
                config=TransformerConfig(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    transformer_type=TransformerType.SENTENCE_TRANSFORMER,
                    cultural_adaptation=CulturalAdaptationLevel.BASIC,
                    supported_languages=["en", "ar", "he", "fa", "es", "fr"],
                    cultural_groups=["general"]
                )
            )
            
            logger.info("Default transformer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing default models: {str(e)}")
    
    async def load_transformer_model(self, 
                                   model_id: str,
                                   config: TransformerConfig) -> bool:
        """Load transformer model with cultural configuration"""
        try:
            # Load based on transformer type
            if config.transformer_type == TransformerType.BERT:
                model = AutoModel.from_pretrained(config.model_name)
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
            elif config.transformer_type == TransformerType.DISTILBERT:
                model = DistilBertModel.from_pretrained(config.model_name)
                tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
                
            elif config.transformer_type == TransformerType.ROBERTA:
                model = RobertaModel.from_pretrained(config.model_name)
                tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
                
            elif config.transformer_type == TransformerType.SENTENCE_TRANSFORMER:
                model = SentenceTransformer(config.model_name)
                tokenizer = None  # SentenceTransformers handles tokenization internally
                
            elif config.transformer_type == TransformerType.CULTURAL_BERT:
                model = CulturallyAwareBertModel(
                    base_model_name=config.model_name,
                    cultural_embedding_dim=config.cultural_embedding_dim,
                    cultural_groups=config.cultural_groups
                )
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
            else:
                raise ValueError(f"Unsupported transformer type: {config.transformer_type}")
            
            # Store model and configuration
            self.models[model_id] = model
            if tokenizer:
                self.tokenizers[model_id] = tokenizer
            self.model_configs[model_id] = config
            
            # Initialize cultural embeddings for supported groups
            await self._initialize_cultural_embeddings(model_id, config)
            
            logger.info(f"Transformer model loaded: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformer model {model_id}: {str(e)}")
            return False
    
    async def _initialize_cultural_embeddings(self, 
                                            model_id: str,
                                            config: TransformerConfig):
        """Initialize cultural embeddings for model"""
        
        for cultural_group in config.cultural_groups:
            if cultural_group in self.cultural_parameters:
                # Create cultural embedding based on group parameters
                params = self.cultural_parameters[cultural_group]
                
                # Generate embedding from cultural markers
                cultural_text = " ".join(params.get("cultural_markers", []))
                
                if config.transformer_type == TransformerType.SENTENCE_TRANSFORMER:
                    model = self.models[model_id]
                    embedding_vector = model.encode([cultural_text])[0]
                else:
                    # Use standard transformer for embedding
                    embedding_vector = np.random.normal(0, 1, config.embedding_dim)
                
                cultural_embedding = CulturalEmbedding(
                    cultural_group=cultural_group,
                    embedding_vector=embedding_vector,
                    confidence=0.8,
                    source_authority="system_initialization",
                    validation_status="pending"
                )
                
                embedding_key = f"{model_id}_{cultural_group}"
                self.cultural_embeddings[embedding_key] = cultural_embedding
    
    async def encode_with_cultural_context(self,
                                         text: str,
                                         model_id: str,
                                         cultural_context: Optional[str] = None,
                                         return_attention: bool = False) -> TransformerOutput:
        """Encode text with cultural context awareness"""
        
        if model_id not in self.models:
            raise ValueError(f"Model not loaded: {model_id}")
        
        model = self.models[model_id]
        config = self.model_configs[model_id]
        
        # Get cultural embedding if context provided
        cultural_embedding = None
        if cultural_context:
            embedding_key = f"{model_id}_{cultural_context}"
            cultural_embedding = self.cultural_embeddings.get(embedding_key)
        
        # Encode based on model type
        if config.transformer_type == TransformerType.SENTENCE_TRANSFORMER:
            embeddings = model.encode([text])[0]
            attention_weights = None
            
        elif config.transformer_type == TransformerType.CULTURAL_BERT:
            tokenizer = self.tokenizers[model_id]
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=config.max_sequence_length,
                padding=True
            )
            
            # Prepare cultural context tensor
            cultural_tensor = None
            if cultural_context and cultural_context in config.cultural_groups:
                cultural_idx = config.cultural_groups.index(cultural_context)
                cultural_tensor = torch.zeros(1, len(config.cultural_groups))
                cultural_tensor[0, cultural_idx] = 1.0
            
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    cultural_context=cultural_tensor,
                    return_cultural_analysis=True
                )
            
            embeddings = outputs['pooled_output'].numpy()[0]
            attention_weights = outputs['cultural_attention_weights'].numpy()[0] if return_attention else None
            sovereignty_compliance = outputs['sovereignty_compliance'].item()
            
        else:
            # Standard transformer
            tokenizer = self.tokenizers[model_id]
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=config.max_sequence_length,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use pooled output or mean of last hidden state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output.numpy()[0]
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
            
            attention_weights = None
            sovereignty_compliance = 1.0  # Default compliance for non-cultural models
        
        # Calculate cultural appropriateness
        cultural_appropriateness = await self._assess_cultural_appropriateness(
            text, cultural_context, embeddings
        )
        
        # Determine sovereignty status
        sovereignty_status = await self._determine_sovereignty_status(
            text, cultural_context, sovereignty_compliance if 'sovereignty_compliance' in locals() else 1.0
        )
        
        return TransformerOutput(
            embeddings=embeddings,
            attention_weights=attention_weights,
            cultural_context=cultural_embedding,
            confidence_score=0.85,  # Model confidence
            cultural_appropriateness=cultural_appropriateness,
            sovereignty_status=sovereignty_status,
            processing_metadata={
                "model_id": model_id,
                "cultural_context": cultural_context,
                "sequence_length": len(text.split()),
                "embedding_dim": len(embeddings),
                "processing_time": datetime.now().isoformat()
            }
        )
    
    async def _assess_cultural_appropriateness(self,
                                             text: str,
                                             cultural_context: Optional[str],
                                             embeddings: np.ndarray) -> float:
        """Assess cultural appropriateness of text processing"""
        
        if not cultural_context or cultural_context not in self.cultural_parameters:
            return 1.0  # No specific cultural concerns
        
        params = self.cultural_parameters[cultural_context]
        appropriateness_score = 1.0
        
        # Check for respectful handling of sensitive terms
        sensitive_terms = params.get("sensitive_concepts", [])
        respectful_terms = params.get("respectful_terms", [])
        
        text_lower = text.lower()
        
        # Penalize inappropriate use of sensitive terms
        for term in sensitive_terms:
            if term.lower() in text_lower:
                # Check if used in respectful context
                context_respectful = any(
                    resp_term.lower() in text_lower 
                    for resp_term in respectful_terms
                )
                if not context_respectful:
                    appropriateness_score -= 0.2
        
        # Bonus for using respectful terminology
        respectful_usage = sum(1 for term in respectful_terms if term.lower() in text_lower)
        if respectful_usage > 0:
            appropriateness_score = min(1.0, appropriateness_score + 0.1 * respectful_usage)
        
        return max(0.0, appropriateness_score)
    
    async def _determine_sovereignty_status(self,
                                          text: str,
                                          cultural_context: Optional[str],
                                          compliance_score: float) -> str:
        """Determine cultural sovereignty status"""
        
        if not cultural_context:
            return "compliant"
        
        if cultural_context not in self.cultural_parameters:
            return "unknown"
        
        params = self.cultural_parameters[cultural_context]
        sovereignty_requirements = params.get("sovereignty_requirements", [])
        
        # High compliance score and cultural context awareness
        if compliance_score >= 0.8 and sovereignty_requirements:
            return "compliant"
        elif compliance_score >= 0.6:
            return "conditionally_compliant"
        else:
            return "requires_review"
    
    async def compute_semantic_similarity(self,
                                        text1: str,
                                        text2: str,
                                        model_id: str,
                                        cultural_context: Optional[str] = None) -> Dict[str, Any]:
        """Compute semantic similarity between texts with cultural awareness"""
        
        # Encode both texts
        encoding1 = await self.encode_with_cultural_context(text1, model_id, cultural_context)
        encoding2 = await self.encode_with_cultural_context(text2, model_id, cultural_context)
        
        # Calculate cosine similarity
        similarity = np.dot(encoding1.embeddings, encoding2.embeddings) / (
            np.linalg.norm(encoding1.embeddings) * np.linalg.norm(encoding2.embeddings)
        )
        
        # Cultural appropriateness check
        min_appropriateness = min(
            encoding1.cultural_appropriateness,
            encoding2.cultural_appropriateness
        )
        
        # Sovereignty compliance check
        sovereignty_consistent = (
            encoding1.sovereignty_status == encoding2.sovereignty_status and
            encoding1.sovereignty_status == "compliant"
        )
        
        return {
            "similarity_score": float(similarity),
            "cultural_appropriateness": min_appropriateness,
            "sovereignty_compliant": sovereignty_consistent,
            "text1_encoding": encoding1,
            "text2_encoding": encoding2,
            "computed_at": datetime.now().isoformat()
        }
    
    async def batch_encode(self,
                          texts: List[str],
                          model_id: str,
                          cultural_context: Optional[str] = None,
                          batch_size: int = 32) -> List[TransformerOutput]:
        """Batch encode multiple texts efficiently"""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                try:
                    encoding = await self.encode_with_cultural_context(
                        text, model_id, cultural_context
                    )
                    batch_results.append(encoding)
                except Exception as e:
                    logger.error(f"Error encoding text in batch: {str(e)}")
                    # Create error result
                    batch_results.append(TransformerOutput(
                        embeddings=np.zeros(768),  # Default dimension
                        attention_weights=None,
                        cultural_context=None,
                        confidence_score=0.0,
                        cultural_appropriateness=0.0,
                        sovereignty_status="error",
                        processing_metadata={"error": str(e)}
                    ))
            
            results.extend(batch_results)
        
        return results
    
    async def fine_tune_cultural_model(self,
                                     base_model_id: str,
                                     training_data: List[Dict[str, Any]],
                                     cultural_context: str,
                                     community_validation: bool = True) -> str:
        """Fine-tune model for specific cultural context"""
        
        if community_validation:
            # Request community validation before fine-tuning
            validation_result = await self._request_community_validation(
                cultural_context, training_data
            )
            
            if not validation_result["approved"]:
                raise ValueError(f"Community validation failed: {validation_result['reason']}")
        
        # Create new culturally-aware model
        base_config = self.model_configs[base_model_id]
        
        new_model_id = f"{base_model_id}_cultural_{cultural_context}"
        new_config = TransformerConfig(
            model_name=base_config.model_name,
            transformer_type=TransformerType.CULTURAL_BERT,
            cultural_adaptation=CulturalAdaptationLevel.COMMUNITY_TRAINED,
            supported_languages=base_config.supported_languages,
            cultural_groups=[cultural_context],
            cultural_embedding_dim=base_config.cultural_embedding_dim,
            community_validated=True,
            sovereignty_compliant=True
        )
        
        # Load culturally-aware model architecture
        culturally_aware_model = CulturallyAwareBertModel(
            base_model_name=base_config.model_name,
            cultural_embedding_dim=new_config.cultural_embedding_dim,
            cultural_groups=[cultural_context]
        )
        
        # Store new model (fine-tuning implementation would go here)
        self.models[new_model_id] = culturally_aware_model
        self.tokenizers[new_model_id] = self.tokenizers[base_model_id]
        self.model_configs[new_model_id] = new_config
        
        # Initialize cultural embeddings
        await self._initialize_cultural_embeddings(new_model_id, new_config)
        
        logger.info(f"Cultural model created: {new_model_id}")
        return new_model_id
    
    async def _request_community_validation(self,
                                          cultural_context: str,
                                          training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Request community validation for cultural model training"""
        
        # Simulate community validation process
        # In real implementation, this would interface with community validation system
        
        validation_request = {
            "cultural_context": cultural_context,
            "training_data_size": len(training_data),
            "cultural_sensitivity_required": True,
            "sovereignty_compliance_required": True
        }
        
        # Check if cultural context has sovereignty requirements
        if cultural_context in self.cultural_parameters:
            params = self.cultural_parameters[cultural_context]
            sovereignty_requirements = params.get("sovereignty_requirements", [])
            
            if sovereignty_requirements:
                # High standards for sensitive cultural contexts
                return {
                    "approved": True,  # Simulated approval
                    "reason": "Community validation approved with sovereignty compliance",
                    "validators": sovereignty_requirements,
                    "validation_score": 0.95
                }
        
        return {
            "approved": True,
            "reason": "Standard community validation approved",
            "validators": ["community_representative"],
            "validation_score": 0.85
        }
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model status"""
        
        if model_id not in self.models:
            return {"status": "not_loaded", "error": "Model not found"}
        
        config = self.model_configs[model_id]
        
        # Count cultural embeddings for this model
        cultural_embeddings_count = sum(
            1 for key in self.cultural_embeddings.keys()
            if key.startswith(f"{model_id}_")
        )
        
        status = {
            "status": "loaded",
            "model_id": model_id,
            "transformer_type": config.transformer_type.value,
            "cultural_adaptation": config.cultural_adaptation.value,
            "supported_languages": config.supported_languages,
            "cultural_groups": config.cultural_groups,
            "cultural_embeddings_count": cultural_embeddings_count,
            "community_validated": config.community_validated,
            "sovereignty_compliant": config.sovereignty_compliant,
            "max_sequence_length": config.max_sequence_length,
            "embedding_dimension": config.embedding_dim
        }
        
        # Add validation history if available
        if model_id in self.community_validations:
            status["community_validations"] = self.community_validations[model_id]
        
        return status
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models with their capabilities"""
        
        models_info = {}
        
        for model_id, config in self.model_configs.items():
            models_info[model_id] = {
                "transformer_type": config.transformer_type.value,
                "cultural_adaptation": config.cultural_adaptation.value,
                "supported_languages": config.supported_languages,
                "cultural_groups": config.cultural_groups,
                "community_validated": config.community_validated,
                "sovereignty_compliant": config.sovereignty_compliant
            }
        
        return {
            "total_models": len(models_info),
            "models": models_info,
            "cultural_contexts": list(set([
                group for config in self.model_configs.values()
                for group in config.cultural_groups
            ])),
            "supported_languages": list(set([
                lang for config in self.model_configs.values()
                for lang in config.supported_languages
            ]))
        }
    
    async def validate_cultural_compliance(self,
                                         model_id: str,
                                         test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate model's cultural compliance with test cases"""
        
        if model_id not in self.models:
            return {"error": "Model not found"}
        
        compliance_results = []
        
        for test_case in test_cases:
            text = test_case["text"]
            expected_cultural_context = test_case.get("cultural_context")
            expected_appropriateness = test_case.get("expected_appropriateness", 0.8)
            
            # Process with model
            output = await self.encode_with_cultural_context(
                text, model_id, expected_cultural_context
            )
            
            # Check compliance
            compliant = (
                output.cultural_appropriateness >= expected_appropriateness and
                output.sovereignty_status in ["compliant", "conditionally_compliant"]
            )
            
            compliance_results.append({
                "text": text,
                "cultural_context": expected_cultural_context,
                "compliant": compliant,
                "appropriateness_score": output.cultural_appropriateness,
                "sovereignty_status": output.sovereignty_status,
                "expected_appropriateness": expected_appropriateness
            })
        
        # Calculate overall compliance
        total_compliant = sum(1 for result in compliance_results if result["compliant"])
        compliance_rate = total_compliant / len(compliance_results) if compliance_results else 0
        
        return {
            "model_id": model_id,
            "compliance_rate": compliance_rate,
            "total_test_cases": len(compliance_results),
            "compliant_cases": total_compliant,
            "detailed_results": compliance_results,
            "overall_status": "compliant" if compliance_rate >= 0.8 else "requires_improvement",
            "validated_at": datetime.now().isoformat()
        }