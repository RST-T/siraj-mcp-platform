"""
Configuration settings for Siraj MCP Server
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class SirajMCPSettings(BaseSettings):
    """
    Configuration settings for Siraj Computational Linguistics MCP Server
    """
    
    # Server Configuration
    server_name: str = Field(default="Siraj-Computational-Linguistics-MCP", description="MCP server name")
    server_version: str = Field(default="6.0.0", description="Server version")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=3001, description="Server port")
    
    # Computational Linguistics Configuration
    max_text_length: int = Field(default=10000, description="Maximum text length for processing")
    max_batch_size: int = Field(default=32, description="Maximum batch size for processing")
    default_language: str = Field(default="en", description="Default language code")
    supported_languages: List[str] = Field(
        default=["en", "ar", "he", "es", "fr", "de", "zh", "ja", "ko", "ru"],
        description="Supported language codes"
    )
    
    # Siraj Framework Configuration
    framework_version: str = Field(default="6.0", description="Siraj framework version")
    max_recursive_loops: int = Field(default=3, description="Maximum recursive analysis loops")
    convergence_threshold: float = Field(default=0.1, description="Convergence threshold for loops")
    significance_level: float = Field(default=0.05, description="Statistical significance level")
    
    # Evidence Weighting Configuration
    primary_evidence_weight: float = Field(default=0.45, description="Primary evidence weight")
    secondary_evidence_weight: float = Field(default=0.30, description="Secondary evidence weight")
    consensus_evidence_weight: float = Field(default=0.15, description="Consensus evidence weight")
    coherence_evidence_weight: float = Field(default=0.10, description="Coherence evidence weight")
    
    # Confidence Thresholds
    text_mode_threshold: float = Field(default=0.75, description="Confidence threshold for text mode")
    psyche_mode_threshold: float = Field(default=0.60, description="Confidence threshold for psyche mode")
    edge_similarity_threshold: float = Field(default=0.65, description="Threshold for graph edge creation")
    
    # Transformer Model Configuration
    multilingual_bert_model: str = Field(
        default="bert-base-multilingual-cased",
        description="Multilingual BERT model name"
    )
    xlm_roberta_model: str = Field(
        default="xlm-roberta-base",
        description="XLM-RoBERTa model name"
    )
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    
    # Model Storage Configuration
    models_cache_dir: str = Field(default="./models/cache", description="Models cache directory")
    models_local_dir: str = Field(default="./models/local", description="Local models directory")
    enable_model_caching: bool = Field(default=True, description="Enable model caching")
    
    # Database Configuration
    corpus_database_url: str = Field(
        default="postgresql://neondb_owner:npg_npWHsoRb5f6v@ep-icy-wave-a8h14f5w-pooler.eastus2.azure.neon.tech/neondb?sslmode=require&channel_binding=require",
        description="Corpus database URL"
    )
    lexicon_database_url: str = Field(
        default="sqlite:///./data/lexicons.db",
        description="Lexicon database URL"
    )
    cache_database_url: str = Field(
        default="redis://localhost:6379/0",
        description="Cache database URL"
    )
    
    # Performance Configuration
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration")
    max_memory_usage: int = Field(default=2 * 1024 * 1024 * 1024, description="Max memory usage (bytes)")
    request_timeout: int = Field(default=120, description="Request timeout in seconds")
    concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    
    # Security Configuration
    enable_authentication: bool = Field(default=True, description="Enable authentication")
    api_key_required: bool = Field(default=False, description="Require API key")
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "https://localhost:3000"],
        description="Allowed CORS origins"
    )
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    
    # HTTPS Configuration
    enable_https: bool = Field(default=False, description="Enable HTTPS server")
    ssl_cert_file: Optional[str] = Field(default=None, description="SSL certificate file path")
    ssl_key_file: Optional[str] = Field(default=None, description="SSL private key file path")
    ssl_ca_file: Optional[str] = Field(default=None, description="SSL CA certificate file path")
    ssl_require_client_cert: bool = Field(default=False, description="Require client certificate")
    valid_api_keys: List[str] = Field(default=["siraj_demo_key_2024"], description="Valid API keys for authentication")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="./logs/siraj_mcp.log", description="Log file path")
    enable_request_logging: bool = Field(default=True, description="Enable request logging")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    
    # Archaeological Semantic Mapping Configuration
    archetypal_nodes_count: int = Field(default=72, description="Number of archetypal nodes")
    tier_weights: Dict[int, float] = Field(
        default={1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1},
        description="Tier importance weights for archetypal mapping"
    )
    node_embedding_dimension: int = Field(default=128, description="Node embedding dimension")
    
    # Cross-Linguistic Configuration
    language_family_mapping: Dict[str, str] = Field(
        default={
            "ar": "semitic", "he": "semitic", "am": "semitic",
            "en": "indo_european", "de": "indo_european", "es": "indo_european",
            "fr": "indo_european", "it": "indo_european", "ru": "indo_european",
            "zh": "sino_tibetan", "my": "sino_tibetan", "bo": "sino_tibetan",
            "fi": "agglutinative", "tr": "agglutinative", "hu": "agglutinative",
            "iu": "polysynthetic", "moh": "polysynthetic",
            "vi": "tonal", "th": "tonal"
        },
        description="Language to language family mapping"
    )
    
    # Statistical Validation Configuration
    bootstrap_samples: int = Field(default=1000, description="Bootstrap sample count")
    permutation_tests: int = Field(default=10000, description="Permutation test iterations")
    confidence_level: float = Field(default=0.95, description="Statistical confidence level")
    
    # External API Configuration
    enable_external_apis: bool = Field(default=True, description="Enable external API calls")
    wordnet_api_url: str = Field(
        default="http://wordnet-api.org/api/v1",
        description="WordNet API base URL"
    )
    external_api_timeout: int = Field(default=30, description="External API timeout")
    external_api_retries: int = Field(default=3, description="External API retry attempts")
    
    # Data Storage Configuration
    data_directory: str = Field(default="./data", description="Data storage directory")
    corpus_data_directory: str = Field(default="./data/corpora", description="Corpus data directory")
    lexicon_data_directory: str = Field(default="./data/lexicons", description="Lexicon data directory")
    output_directory: str = Field(default="./output", description="Output directory")
    
    # Validation and Testing
    validation_corpus_path: str = Field(
        default="./data/validation/validation_corpus.json",
        description="Validation corpus path"
    )
    test_corpus_path: str = Field(
        default="./data/validation/test_corpus.json",
        description="Test corpus path"
    )
    enable_validation_on_startup: bool = Field(
        default=True,
        description="Run validation tests on startup"
    )
    
    @validator("primary_evidence_weight", "secondary_evidence_weight", 
              "consensus_evidence_weight", "coherence_evidence_weight")
    def validate_evidence_weights(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Evidence weights must be between 0.0 and 1.0")
        return v
    
    @validator("text_mode_threshold", "psyche_mode_threshold", "edge_similarity_threshold")
    def validate_thresholds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v
    
    @validator("significance_level", "confidence_level")
    def validate_statistical_levels(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("Statistical levels must be between 0.0 and 1.0")
        return v
    
    def validate_evidence_weights_sum(self) -> None:
        """Validate that evidence weights sum to 1.0"""
        total = (
            self.primary_evidence_weight +
            self.secondary_evidence_weight +
            self.consensus_evidence_weight +
            self.coherence_evidence_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Evidence weights must sum to 1.0, got {total}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get transformer model configuration"""
        return {
            "multilingual_bert": {
                "model_name": self.multilingual_bert_model,
                "cache_dir": self.models_cache_dir,
                "local_files_only": False
            },
            "xlm_roberta": {
                "model_name": self.xlm_roberta_model,
                "cache_dir": self.models_cache_dir,
                "local_files_only": False
            },
            "sentence_transformer": {
                "model_name": self.sentence_transformer_model,
                "cache_dir": self.models_cache_dir,
                "local_files_only": False
            }
        }
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration"""
        return {
            "corpus": self.corpus_database_url,
            "lexicon": self.lexicon_database_url,
            "cache": self.cache_database_url
        }
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        directories = [
            self.models_cache_dir,
            self.models_local_dir,
            self.data_directory,
            self.corpus_data_directory,
            self.lexicon_data_directory,
            self.output_directory,
            os.path.dirname(self.log_file) if self.log_file else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_prefix = "SIRAJ_"
        case_sensitive = False


# Global settings instance
settings = SirajMCPSettings()

# Validate settings on import
settings.validate_evidence_weights_sum()
settings.ensure_directories()