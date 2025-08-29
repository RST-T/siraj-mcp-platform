"""
Linguistic data types and models for Siraj MCP Server
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np
from datetime import datetime


class LanguageFamily(str, Enum):
    """Language family enumeration"""
    SEMITIC = "semitic"
    INDO_EUROPEAN = "indo_european"
    SINO_TIBETAN = "sino_tibetan"
    AGGLUTINATIVE = "agglutinative"
    POLYSYNTHETIC = "polysynthetic"
    TONAL = "tonal"
    UNKNOWN = "unknown"


class AnalysisMode(str, Enum):
    """Analysis mode enumeration"""
    TEXT = "text"
    PSYCHE = "psyche"


class ArchetypalTier(int, Enum):
    """Archetypal node tier enumeration"""
    CORE_PRIMITIVES = 1
    BASIC_FORCES = 2
    LINGUISTIC_DYNAMICS = 3
    CONTEXTUAL_MATRIX = 4
    COMPLEX_INTEGRATION = 5


class EvidenceType(str, Enum):
    """Evidence type enumeration"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONSENSUS = "consensus"
    COHERENCE = "coherence"


class LinguisticFeature(BaseModel):
    """Base linguistic feature representation"""
    feature_type: str = Field(description="Type of linguistic feature")
    value: Any = Field(description="Feature value")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(description="Source of the feature")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class MorphologicalAnalysis(BaseModel):
    """Morphological analysis results"""
    word: str = Field(description="Input word")
    language: str = Field(description="Language code")
    morphemes: List[Dict[str, Any]] = Field(description="Identified morphemes")
    features: Dict[str, Any] = Field(description="Morphological features")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    algorithm: str = Field(description="Analysis algorithm used")
    processing_time: float = Field(description="Processing time in seconds")


class SemanticNode(BaseModel):
    """Archetypal semantic node representation"""
    node_id: int = Field(ge=1, le=72, description="Node ID (1-72)")
    tier: ArchetypalTier = Field(description="Archetypal tier")
    name: str = Field(description="Node name")
    proto_semitic_root: str = Field(description="Proto-Semitic root")
    computational_function: str = Field(description="Computational linguistic function")
    linguistic_application: str = Field(description="Linguistic application description")
    confidence: float = Field(ge=0.0, le=1.0, description="Mapping confidence")


class TransformerAnalysis(BaseModel):
    """Transformer model analysis results"""
    model_name: str = Field(description="Transformer model name")
    embeddings: List[float] = Field(description="Generated embeddings")
    attention_patterns: Optional[Dict[str, Any]] = Field(default=None, description="Attention patterns")
    tokens: List[str] = Field(description="Tokenized input")
    token_count: int = Field(description="Number of tokens")
    processing_time: float = Field(description="Processing time in seconds")


class EvidenceResult(BaseModel):
    """Evidence calculation results"""
    evidence_type: EvidenceType = Field(description="Type of evidence")
    value: float = Field(ge=0.0, le=1.0, description="Evidence value")
    sources: List[str] = Field(description="Evidence sources")
    metadata: Dict[str, Any] = Field(description="Evidence metadata")
    statistical_validation: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistical validation results"
    )


class SemanticMapping(BaseModel):
    """Semantic mapping results"""
    primary_node: SemanticNode = Field(description="Primary mapped node")
    alternative_nodes: List[SemanticNode] = Field(description="Alternative node mappings")
    mapping_confidence: float = Field(ge=0.0, le=1.0, description="Overall mapping confidence")
    hierarchical_path: List[int] = Field(description="Hierarchical classification path")
    semantic_features: Dict[str, Any] = Field(description="Extracted semantic features")


class CrossLinguisticAnalysis(BaseModel):
    """Cross-linguistic analysis results"""
    source_language: str = Field(description="Source language code")
    source_family: LanguageFamily = Field(description="Source language family")
    target_projections: Dict[str, Dict[str, Any]] = Field(description="Target language projections")
    universal_features: Dict[str, Any] = Field(description="Universal linguistic features")
    cross_coherence: float = Field(ge=0.0, le=1.0, description="Cross-linguistic coherence score")
    projection_confidence: Dict[str, float] = Field(description="Projection confidence scores")


class SemanticGraph(BaseModel):
    """Semantic graph representation"""
    nodes: List[Dict[str, Any]] = Field(description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(description="Graph edges")
    graph_metrics: Dict[str, Any] = Field(description="Graph analysis metrics")
    validation_metrics: Dict[str, Any] = Field(description="Graph validation results")


class SemanticDrift(BaseModel):
    """Semantic drift analysis results"""
    source_context: str = Field(description="Source context name")
    target_context: str = Field(description="Target context name")
    drift_magnitude: float = Field(ge=0.0, description="Magnitude of semantic drift")
    drift_direction: List[float] = Field(description="Direction of drift in embedding space")
    dominant_changes: List[Dict[str, Any]] = Field(description="Most significant changes")
    statistical_significance: Dict[str, Any] = Field(description="Statistical significance tests")
    convergence_analysis: Optional[Dict[str, Any]] = Field(
        default=None, description="Convergence analysis results"
    )


class ValidationMetrics(BaseModel):
    """Validation and quality metrics"""
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Accuracy score")
    precision_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Precision score")
    recall_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Recall score")
    f1_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="F1 score")
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None, description="Confidence interval"
    )
    statistical_significance: Optional[Dict[str, Any]] = Field(
        default=None, description="Statistical significance results"
    )
    reproducibility_hash: Optional[str] = Field(
        default=None, description="Reproducibility hash"
    )


class ContextAnalysis(BaseModel):
    """Complete context analysis results"""
    context_name: str = Field(description="Context identifier")
    processed_data: Dict[str, Any] = Field(description="Processed input data")
    evidence_results: List[EvidenceResult] = Field(description="Evidence calculation results")
    combined_confidence: float = Field(ge=0.0, le=1.0, description="Combined evidence confidence")
    semantic_mapping: Optional[SemanticMapping] = Field(
        default=None, description="Semantic mapping results"
    )
    cross_linguistic_analysis: Optional[CrossLinguisticAnalysis] = Field(
        default=None, description="Cross-linguistic analysis"
    )
    semantic_graph: Optional[SemanticGraph] = Field(default=None, description="Semantic graph")
    validation_metrics: ValidationMetrics = Field(description="Validation metrics")
    processing_metadata: Dict[str, Any] = Field(description="Processing metadata")


class SirajAnalysisRequest(BaseModel):
    """Siraj analysis request model"""
    root: str = Field(description="Etymological root for analysis")
    contexts: Dict[str, Any] = Field(description="Contexts to analyze")
    mode: AnalysisMode = Field(default=AnalysisMode.TEXT, description="Analysis mode")
    language_family: Union[LanguageFamily, str] = Field(
        default="auto_detect", description="Target language family"
    )
    analysis_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Analysis configuration"
    )
    
    @validator("contexts")
    def validate_contexts(cls, v):
        if not v:
            raise ValueError("At least one context must be provided")
        return v


class SirajAnalysisResponse(BaseModel):
    """Siraj analysis response model"""
    framework_version: str = Field(description="Siraj framework version")
    root: str = Field(description="Analyzed root")
    mode: AnalysisMode = Field(description="Analysis mode used")
    language_family: LanguageFamily = Field(description="Detected/specified language family")
    timestamp: datetime = Field(description="Analysis timestamp")
    
    context_analyses: Dict[str, ContextAnalysis] = Field(description="Context analysis results")
    semantic_drift_analysis: Optional[List[SemanticDrift]] = Field(
        default=None, description="Cross-context semantic drift analysis"
    )
    global_semantic_integration: Optional[Dict[str, Any]] = Field(
        default=None, description="Global semantic integration results"
    )
    
    # Overall metrics
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall analysis confidence")
    processing_time: float = Field(description="Total processing time in seconds")
    validation_report: ValidationMetrics = Field(description="Comprehensive validation report")
    
    # Metadata
    framework_metadata: Dict[str, Any] = Field(description="Framework metadata")
    scientific_metadata: Dict[str, Any] = Field(description="Scientific validation metadata")
    reproducibility_info: Dict[str, Any] = Field(description="Reproducibility information")


class MCPToolResponse(BaseModel):
    """Base MCP tool response model"""
    success: bool = Field(description="Whether the operation was successful")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class MCPResourceResponse(BaseModel):
    """Base MCP resource response model"""
    resource_type: str = Field(description="Type of resource")
    resource_id: str = Field(description="Resource identifier")
    content: Any = Field(description="Resource content")
    metadata: Dict[str, Any] = Field(description="Resource metadata")
    last_modified: Optional[datetime] = Field(default=None, description="Last modification time")
    access_time: datetime = Field(default_factory=datetime.now, description="Access time")


# Utility type aliases
EmbeddingVector = List[float]
ConfidenceScore = float
LanguageCode = str
ContextName = str
NodeID = int
ProcessingTime = float