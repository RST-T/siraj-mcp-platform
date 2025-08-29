"""
Comprehensive Testing Framework for SIRAJ v6.1
Implements multi-paradigm testing across traditional, scientific, and computational domains
"""

import pytest
import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import logging

# Import SIRAJ components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from server.siraj_v6_1_engine import SirajV61ComputationalHermeneuticsEngine
from server.adaptive_semantic_architecture import AdaptiveSemanticArchitecture
from server.community_validation_interface import CommunityValidationInterface, ValidationLevel, ValidatorRole
from server.cultural_sovereignty_manager import CulturalSovereigntyManager, CulturalSensitivityLevel
from server.multi_paradigm_validator import MultiParadigmValidator, ValidationParadigm
from server.transformer_integration import CommunityInformedTransformerManager, TransformerType
from server.performance_optimizer import PerformanceOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)

class SirajTestFramework:
    """
    Comprehensive testing framework for SIRAJ v6.1 system
    """
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.mock_config = self._create_mock_config()
        self.test_data = self._load_test_data()
        
    def _create_mock_config(self) -> Dict[str, Any]:
        """Create mock configuration for testing"""
        return {
            "cultural_sovereignty": {
                "enable_cultural_boundaries": True,
                "require_community_validation": True,
                "cultural_sensitivity_threshold": 0.8
            },
            "validation": {
                "multi_paradigm_enabled": True,
                "traditional_weight": 1.5,
                "scientific_weight": 1.2,
                "computational_weight": 1.0,
                "community_weight": 1.3
            },
            "transformer": {
                "default_model": "multilingual_bert",
                "cultural_adaptation": True,
                "batch_size": 16
            },
            "performance": {
                "enable_caching": True,
                "enable_batching": True,
                "max_cache_size": 1000
            }
        }
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for various scenarios"""
        return {
            "arabic_religious_texts": [
                {
                    "text": "بسم الله الرحمن الرحيم",
                    "translation": "In the name of Allah, the Most Gracious, the Most Merciful",
                    "cultural_context": "islamic",
                    "sensitivity_level": CulturalSensitivityLevel.SACRED,
                    "expected_root": "سمو",
                    "linguistic_family": "semitic"
                },
                {
                    "text": "الحمد لله رب العالمين",
                    "translation": "Praise be to Allah, Lord of the worlds",
                    "cultural_context": "islamic",
                    "sensitivity_level": CulturalSensitivityLevel.SACRED,
                    "expected_root": "حمد",
                    "linguistic_family": "semitic"
                }
            ],
            "comparative_linguistic_data": [
                {
                    "arabic_root": "كتب",
                    "hebrew_cognate": "כתב",
                    "semantic_field": "writing",
                    "proto_semitic": "*katab-",
                    "cultural_context": "semitic_linguistic"
                },
                {
                    "arabic_root": "قدس",
                    "hebrew_cognate": "קדש",
                    "semantic_field": "sanctity",
                    "proto_semitic": "*qadas-",
                    "cultural_context": "semitic_linguistic"
                }
            ],
            "cultural_boundary_tests": [
                {
                    "content": "Analysis of Quranic interpretation methods",
                    "cultural_context": "islamic",
                    "expected_permission": "traditional_authority",
                    "should_pass": False  # Without proper permissions
                },
                {
                    "content": "General Arabic linguistic patterns",
                    "cultural_context": "semitic_linguistic",
                    "expected_permission": "community_member",
                    "should_pass": True
                }
            ],
            "performance_test_cases": [
                {
                    "batch_size": 1,
                    "expected_cache_hits": 0,
                    "texts": ["Test text for performance analysis"]
                },
                {
                    "batch_size": 10,
                    "expected_cache_hits": 5,
                    "texts": [f"Performance test {i}" for i in range(10)]
                }
            ]
        }

class TestSirajV61Engine(unittest.TestCase):
    """Test cases for SIRAJ v6.1 Computational Hermeneutics Engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.engine = SirajV61ComputationalHermeneuticsEngine(self.framework.mock_config)
    
    @pytest.mark.asyncio
    async def test_enhanced_semantic_reconstruction(self):
        """Test enhanced semantic reconstruction pipeline"""
        
        test_case = self.framework.test_data["arabic_religious_texts"][0]
        
        result = await self.engine.enhanced_semantic_reconstruction(
            root=test_case["expected_root"],
            contexts={
                "cultural_context": test_case["cultural_context"],
                "text": test_case["text"],
                "translation": test_case["translation"]
            },
            mode="text",
            language_family="semitic"
        )
        
        # Validate result structure
        self.assertIn("root_analysis", result)
        self.assertIn("cultural_context", result)
        self.assertIn("semantic_reconstruction", result)
        self.assertIn("validation_results", result)
        
        # Check cultural sensitivity
        cultural_result = result.get("cultural_context", {})
        self.assertGreaterEqual(cultural_result.get("appropriateness_score", 0), 0.8)
        
        # Verify linguistic analysis
        root_analysis = result.get("root_analysis", {})
        self.assertEqual(root_analysis.get("primary_root"), test_case["expected_root"])
        
        logger.info(f"Enhanced semantic reconstruction test passed: {test_case['text']}")
    
    @pytest.mark.asyncio
    async def test_comparative_analysis(self):
        """Test comparative linguistic analysis"""
        
        test_case = self.framework.test_data["comparative_linguistic_data"][0]
        
        result = await self.engine.perform_comparative_analysis(
            arabic_root=test_case["arabic_root"],
            comparative_data={
                "hebrew": test_case["hebrew_cognate"],
                "proto_semitic": test_case["proto_semitic"]
            },
            semantic_field=test_case["semantic_field"]
        )
        
        # Validate comparative analysis
        self.assertIn("cognate_relationships", result)
        self.assertIn("semantic_evolution", result)
        self.assertIn("cultural_implications", result)
        
        # Check cognate detection
        cognates = result.get("cognate_relationships", {})
        self.assertIn("hebrew", cognates)
        self.assertGreaterEqual(cognates.get("confidence_score", 0), 0.7)
        
        logger.info(f"Comparative analysis test passed: {test_case['arabic_root']}")
    
    def test_cultural_sensitivity_detection(self):
        """Test cultural sensitivity detection"""
        
        sensitive_text = "Sacred Quranic verse analysis"
        result = self.engine._assess_cultural_sensitivity(
            sensitive_text,
            "islamic"
        )
        
        self.assertGreater(result["sensitivity_score"], 0.8)
        self.assertEqual(result["sensitivity_level"], "sacred")
        
        logger.info("Cultural sensitivity detection test passed")

class TestAdaptiveSemanticArchitecture(unittest.TestCase):
    """Test cases for Adaptive Semantic Architecture"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.asa = AdaptiveSemanticArchitecture(self.framework.mock_config)
    
    @pytest.mark.asyncio
    async def test_node_generation(self):
        """Test semantic node generation"""
        
        test_case = self.framework.test_data["arabic_religious_texts"][0]
        
        nodes = await self.asa.generate_semantic_nodes(
            content=test_case["text"],
            cultural_context=test_case["cultural_context"],
            tier_level=2
        )
        
        self.assertGreater(len(nodes), 0)
        
        # Check node structure
        for node in nodes:
            self.assertIn("node_id", node)
            self.assertIn("semantic_content", node)
            self.assertIn("cultural_markers", node)
            self.assertIn("confidence_score", node)
            self.assertGreaterEqual(node["confidence_score"], 0.5)
        
        logger.info(f"Node generation test passed: {len(nodes)} nodes generated")
    
    @pytest.mark.asyncio
    async def test_cultural_adaptation(self):
        """Test cultural adaptation mechanisms"""
        
        adaptation_result = await self.asa.adapt_to_cultural_context(
            "islamic",
            {
                "content_type": "religious",
                "sensitivity_level": "high",
                "community_validation": True
            }
        )
        
        self.assertTrue(adaptation_result["adapted"])
        self.assertIn("cultural_parameters", adaptation_result)
        self.assertIn("adaptation_strategy", adaptation_result)
        
        # Check cultural parameters
        params = adaptation_result["cultural_parameters"]
        self.assertIn("respectful_terminology", params)
        self.assertIn("boundary_conditions", params)
        
        logger.info("Cultural adaptation test passed")
    
    def test_tier_validation(self):
        """Test tier system validation"""
        
        # Test all 5 tiers
        for tier in range(1, 6):
            tier_info = self.asa.get_tier_info(tier)
            
            self.assertIn("tier_level", tier_info)
            self.assertIn("node_capacity", tier_info)
            self.assertIn("adaptation_rate", tier_info)
            self.assertIn("community_influence", tier_info)
        
        logger.info("Tier validation test passed")

class TestCommunityValidationInterface(unittest.TestCase):
    """Test cases for Community Validation Interface"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.validation_interface = CommunityValidationInterface(self.framework.mock_config)
    
    @pytest.mark.asyncio
    async def test_validator_registration(self):
        """Test community validator registration"""
        
        success = await self.validation_interface.register_community_validator(
            validator_id="test_scholar",
            role=ValidatorRole.LINGUISTIC_SCHOLAR,
            credentials={"academic_credentials": "PhD Linguistics"},
            cultural_affiliations=["semitic_linguistic"],
            linguistic_expertise=["arabic", "hebrew"]
        )
        
        self.assertTrue(success)
        self.assertIn("test_scholar", self.validation_interface.community_validators)
        
        validator_data = self.validation_interface.community_validators["test_scholar"]
        self.assertEqual(validator_data["role"], ValidatorRole.LINGUISTIC_SCHOLAR)
        
        logger.info("Validator registration test passed")
    
    @pytest.mark.asyncio
    async def test_validation_request_submission(self):
        """Test validation request submission"""
        
        # Register a validator first
        await self.validation_interface.register_community_validator(
            validator_id="cultural_keeper",
            role=ValidatorRole.CULTURAL_KEEPER,
            credentials={"community_recognition": True},
            cultural_affiliations=["islamic"],
            linguistic_expertise=["arabic"]
        )
        
        test_case = self.framework.test_data["cultural_boundary_tests"][0]
        
        request_id = await self.validation_interface.submit_validation_request(
            content_hash="test_hash_123",
            validation_data={
                "content": test_case["content"],
                "cultural_context": test_case["cultural_context"]
            },
            validation_level=ValidationLevel.COMPREHENSIVE,
            cultural_context=test_case["cultural_context"]
        )
        
        self.assertIsNotNone(request_id)
        self.assertTrue(len(request_id) > 0)
        
        logger.info(f"Validation request submission test passed: {request_id}")
    
    @pytest.mark.asyncio
    async def test_consensus_calculation(self):
        """Test community consensus calculation"""
        
        # Mock some validation results
        content_hash = "test_consensus_hash"
        self.validation_interface.validation_history[content_hash] = []
        
        # Add mock validation results
        from server.community_validation_interface import ValidationResult
        
        for i in range(3):
            result = ValidationResult(
                validator_id=f"validator_{i}",
                validator_role=ValidatorRole.LINGUISTIC_SCHOLAR,
                confidence_score=0.85 + (i * 0.05),
                cultural_appropriateness=0.90,
                linguistic_accuracy=0.88,
                computational_validity=0.82,
                comments=f"Validation {i} looks good"
            )
            self.validation_interface.validation_history[content_hash].append(result)
        
        consensus = await self.validation_interface.calculate_community_consensus(
            content_hash,
            ValidationLevel.COMPREHENSIVE
        )
        
        self.assertIsNotNone(consensus)
        self.assertGreaterEqual(consensus.consensus_score, 0.7)
        self.assertTrue(consensus.cultural_sovereignty_maintained)
        
        logger.info(f"Consensus calculation test passed: {consensus.consensus_score}")

class TestCulturalSovereigntyManager(unittest.TestCase):
    """Test cases for Cultural Sovereignty Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.sovereignty_manager = CulturalSovereigntyManager(self.framework.mock_config)
    
    @pytest.mark.asyncio
    async def test_cultural_context_assessment(self):
        """Test cultural context assessment"""
        
        test_case = self.framework.test_data["arabic_religious_texts"][0]
        
        cultural_context = await self.sovereignty_manager.assess_cultural_context(
            content=test_case["text"],
            metadata={
                "translation": test_case["translation"],
                "cultural_group": test_case["cultural_context"]
            }
        )
        
        self.assertEqual(cultural_context.primary_culture, test_case["cultural_context"])
        self.assertIn("sacred_content", cultural_context.sensitivity_markers)
        self.assertIn("traditional_authority", 
                     [perm.value for perm in cultural_context.required_permissions])
        
        logger.info(f"Cultural context assessment test passed: {cultural_context.primary_culture}")
    
    @pytest.mark.asyncio
    async def test_sovereignty_compliance(self):
        """Test sovereignty compliance checking"""
        
        test_case = self.framework.test_data["cultural_boundary_tests"][0]
        
        # Create mock cultural context
        from server.cultural_sovereignty_manager import CulturalContext, AccessPermissionLevel
        
        cultural_context = CulturalContext(
            primary_culture="islamic",
            secondary_cultures=[],
            linguistic_family="semitic",
            religious_context="islamic",
            geographical_origin=None,
            temporal_period=None,
            sensitivity_markers=["sacred_content"],
            required_permissions=[AccessPermissionLevel.TRADITIONAL_AUTHORITY]
        )
        
        # Test with insufficient permissions
        compliance = await self.sovereignty_manager.check_sovereignty_compliance(
            content_hash="test_compliance",
            cultural_context=cultural_context,
            proposed_action="analysis",
            user_permissions={"islamic": AccessPermissionLevel.COMMUNITY_MEMBER}
        )
        
        self.assertFalse(compliance["compliant"])
        self.assertGreater(len(compliance["violations"]), 0)
        
        logger.info("Sovereignty compliance test passed")
    
    @pytest.mark.asyncio
    async def test_content_protection(self):
        """Test cultural content protection"""
        
        protection_result = await self.sovereignty_manager.protect_cultural_content(
            content_hash="protected_content_123",
            cultural_context=None,  # Will be created in method
            protection_level=CulturalSensitivityLevel.SACRED
        )
        
        self.assertTrue(protection_result["protected"])
        self.assertIn("traditional_authority_required", protection_result["protection_protocols"])
        
        # Check that content is registered
        self.assertIn("protected_content_123", self.sovereignty_manager.protected_content_registry)
        
        logger.info("Content protection test passed")

class TestMultiParadigmValidator(unittest.TestCase):
    """Test cases for Multi-Paradigm Validator"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.validator = MultiParadigmValidator(self.framework.mock_config)
    
    @pytest.mark.asyncio
    async def test_multi_paradigm_validation(self):
        """Test comprehensive multi-paradigm validation"""
        
        test_case = self.framework.test_data["arabic_religious_texts"][0]
        
        validation_report = await self.validator.validate_multi_paradigm(
            content=test_case["text"],
            metadata={
                "cultural_context": test_case["cultural_context"],
                "translation": test_case["translation"],
                "sources": ["Quran"]
            },
            paradigms=[
                ValidationParadigm.TRADITIONAL,
                ValidationParadigm.CULTURAL,
                ValidationParadigm.COMPUTATIONAL
            ]
        )
        
        self.assertIsNotNone(validation_report)
        self.assertIn(ValidationParadigm.TRADITIONAL, validation_report.paradigm_results)
        self.assertIn(ValidationParadigm.CULTURAL, validation_report.paradigm_results)
        self.assertGreaterEqual(validation_report.overall_score, 0.5)
        
        logger.info(f"Multi-paradigm validation test passed: {validation_report.overall_score}")
    
    def test_paradigm_specific_validation(self):
        """Test individual paradigm validations"""
        
        # Test traditional paradigm
        traditional_criteria = self.validator.validation_criteria[ValidationParadigm.TRADITIONAL]
        self.assertEqual(traditional_criteria.success_threshold, 0.8)
        self.assertEqual(traditional_criteria.weight, 1.5)
        
        # Test scientific paradigm
        scientific_criteria = self.validator.validation_criteria[ValidationParadigm.SCIENTIFIC]
        self.assertEqual(scientific_criteria.success_threshold, 0.85)
        self.assertEqual(scientific_criteria.confidence_interval, 0.95)
        
        logger.info("Paradigm-specific validation test passed")
    
    def test_cross_paradigm_consistency(self):
        """Test cross-paradigm consistency calculation"""
        
        # Mock paradigm results
        from server.multi_paradigm_validator import ValidationResult, ValidationMethod
        
        mock_results = {
            ValidationParadigm.TRADITIONAL: [
                ValidationResult(
                    paradigm=ValidationParadigm.TRADITIONAL,
                    method=ValidationMethod.SCHOLARLY_CONSENSUS,
                    success=True,
                    score=0.85,
                    confidence=0.9,
                    evidence={},
                    validator_id="traditional_validator"
                )
            ],
            ValidationParadigm.SCIENTIFIC: [
                ValidationResult(
                    paradigm=ValidationParadigm.SCIENTIFIC,
                    method=ValidationMethod.STATISTICAL_SIGNIFICANCE,
                    success=True,
                    score=0.82,
                    confidence=0.95,
                    evidence={},
                    validator_id="scientific_validator"
                )
            ]
        }
        
        consistency = asyncio.run(
            self.validator._calculate_cross_paradigm_consistency(mock_results)
        )
        
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
        
        logger.info(f"Cross-paradigm consistency test passed: {consistency}")

class TestTransformerIntegration(unittest.TestCase):
    """Test cases for Community-Informed Transformer Integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        # Mock transformer manager to avoid loading actual models in tests
        with patch('server.transformer_integration.AutoModel'), \
             patch('server.transformer_integration.AutoTokenizer'), \
             patch('server.transformer_integration.SentenceTransformer'):
            self.transformer_manager = CommunityInformedTransformerManager(self.framework.mock_config)
    
    @patch('server.transformer_integration.AutoModel.from_pretrained')
    @patch('server.transformer_integration.AutoTokenizer.from_pretrained')
    def test_model_loading(self, mock_tokenizer, mock_model):
        """Test transformer model loading with cultural configuration"""
        
        from server.transformer_integration import TransformerConfig
        
        config = TransformerConfig(
            model_name="bert-base-multilingual-cased",
            transformer_type=TransformerType.BERT,
            cultural_adaptation="intermediate",
            supported_languages=["en", "ar"],
            cultural_groups=["islamic"]
        )
        
        # Mock the model and tokenizer
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        
        success = asyncio.run(
            self.transformer_manager.load_transformer_model("test_model", config)
        )
        
        self.assertTrue(success)
        self.assertIn("test_model", self.transformer_manager.model_configs)
        
        logger.info("Model loading test passed")
    
    def test_cultural_appropriateness_assessment(self):
        """Test cultural appropriateness assessment"""
        
        # Test with respectful Islamic text
        appropriateness_score = asyncio.run(
            self.transformer_manager._assess_cultural_appropriateness(
                "In the name of Allah, the Most Gracious",
                "islamic",
                np.random.normal(0, 1, 768)
            )
        )
        
        self.assertGreaterEqual(appropriateness_score, 0.8)
        
        # Test with potentially inappropriate text
        inappropriateness_score = asyncio.run(
            self.transformer_manager._assess_cultural_appropriateness(
                "Speculative interpretation of divine text",
                "islamic",
                np.random.normal(0, 1, 768)
            )
        )
        
        self.assertLess(inappropriateness_score, 0.9)
        
        logger.info("Cultural appropriateness assessment test passed")
    
    def test_model_status_reporting(self):
        """Test model status reporting"""
        
        # Add a mock model configuration
        from server.transformer_integration import TransformerConfig, CulturalAdaptationLevel
        
        test_config = TransformerConfig(
            model_name="test_model",
            transformer_type=TransformerType.BERT,
            cultural_adaptation=CulturalAdaptationLevel.INTERMEDIATE,
            supported_languages=["en"],
            cultural_groups=["general"]
        )
        
        self.transformer_manager.model_configs["test_model"] = test_config
        self.transformer_manager.models["test_model"] = Mock()
        
        status = self.transformer_manager.get_model_status("test_model")
        
        self.assertEqual(status["status"], "loaded")
        self.assertEqual(status["transformer_type"], "bert")
        self.assertIn("supported_languages", status)
        
        logger.info("Model status reporting test passed")

class TestPerformanceOptimizer(unittest.TestCase):
    """Test cases for Performance Optimizer"""
    
    def setUp(self):
        """Set up test environment"""
        self.framework = SirajTestFramework()
        self.optimizer = PerformanceOptimizer(OptimizationConfig())
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        
        # Test cache put and get
        test_content = "Test content for caching"
        test_result = {"processed": True, "score": 0.85}
        
        self.optimizer._cache_result(
            {"text": test_content, "operation": "test"},
            test_result
        )
        
        cache_key = self.optimizer._generate_cache_key(
            {"text": test_content, "operation": "test"}
        )
        
        cached_result = self.optimizer.cache.get(cache_key)
        self.assertEqual(cached_result, test_result)
        
        logger.info("Cache functionality test passed")
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        
        # Record some performance metrics
        self.optimizer._record_performance(0.5, "test_optimization")
        self.optimizer._record_performance(0.3, "cache_hit")
        
        summary = self.optimizer.get_performance_summary()
        
        self.assertIn("performance_metrics", summary)
        self.assertIn("cache_stats", summary)
        self.assertGreater(len(self.optimizer.performance_history), 0)
        
        logger.info("Performance monitoring test passed")
    
    @pytest.mark.asyncio
    async def test_request_optimization(self):
        """Test request optimization pipeline"""
        
        # Mock processor function
        async def mock_processor(request_data):
            return {"result": "processed", "data": request_data}
        
        request_data = {
            "text": "Test optimization request",
            "operation": "analysis",
            "cultural_context": "general"
        }
        
        result = await self.optimizer.optimize_request(request_data, mock_processor)
        
        self.assertIn("result", result)
        self.assertIn("processing_time", result)
        self.assertIn("optimization", result)
        
        logger.info("Request optimization test passed")

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios combining multiple components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.framework = SirajTestFramework()
        
        # Initialize all components
        self.engine = SirajV61ComputationalHermeneuticsEngine(self.framework.mock_config)
        self.asa = AdaptiveSemanticArchitecture(self.framework.mock_config)
        self.validation_interface = CommunityValidationInterface(self.framework.mock_config)
        self.sovereignty_manager = CulturalSovereigntyManager(self.framework.mock_config)
        self.validator = MultiParadigmValidator(self.framework.mock_config)
        self.optimizer = PerformanceOptimizer(OptimizationConfig())
    
    @pytest.mark.asyncio
    async def test_full_pipeline_islamic_text(self):
        """Test complete pipeline with Islamic religious text"""
        
        test_case = self.framework.test_data["arabic_religious_texts"][0]
        
        # Step 1: Cultural context assessment
        cultural_context = await self.sovereignty_manager.assess_cultural_context(
            content=test_case["text"],
            metadata={"cultural_group": test_case["cultural_context"]}
        )
        
        self.assertEqual(cultural_context.primary_culture, "islamic")
        
        # Step 2: Semantic reconstruction
        reconstruction_result = await self.engine.enhanced_semantic_reconstruction(
            root=test_case["expected_root"],
            contexts={
                "cultural_context": test_case["cultural_context"],
                "text": test_case["text"]
            }
        )
        
        self.assertIn("semantic_reconstruction", reconstruction_result)
        
        # Step 3: Multi-paradigm validation
        validation_report = await self.validator.validate_multi_paradigm(
            content=test_case["text"],
            metadata={"cultural_context": test_case["cultural_context"]}
        )
        
        self.assertTrue(validation_report.overall_validity or validation_report.overall_score > 0.6)
        
        logger.info("Full pipeline Islamic text test passed")
    
    @pytest.mark.asyncio
    async def test_comparative_linguistics_workflow(self):
        """Test comparative linguistics analysis workflow"""
        
        test_case = self.framework.test_data["comparative_linguistic_data"][0]
        
        # Step 1: Generate semantic nodes for both languages
        arabic_nodes = await self.asa.generate_semantic_nodes(
            content=f"Arabic root: {test_case['arabic_root']}",
            cultural_context=test_case["cultural_context"],
            tier_level=2
        )
        
        hebrew_nodes = await self.asa.generate_semantic_nodes(
            content=f"Hebrew cognate: {test_case['hebrew_cognate']}",
            cultural_context=test_case["cultural_context"],
            tier_level=2
        )
        
        self.assertGreater(len(arabic_nodes), 0)
        self.assertGreater(len(hebrew_nodes), 0)
        
        # Step 2: Perform comparative analysis
        comparative_result = await self.engine.perform_comparative_analysis(
            arabic_root=test_case["arabic_root"],
            comparative_data={"hebrew": test_case["hebrew_cognate"]},
            semantic_field=test_case["semantic_field"]
        )
        
        self.assertIn("cognate_relationships", comparative_result)
        
        # Step 3: Validate with multiple paradigms
        validation_report = await self.validator.validate_multi_paradigm(
            content=f"Comparative analysis: {test_case['arabic_root']} <-> {test_case['hebrew_cognate']}",
            metadata={
                "cultural_context": test_case["cultural_context"],
                "analysis_type": "comparative_linguistics"
            }
        )
        
        self.assertGreaterEqual(validation_report.overall_score, 0.5)
        
        logger.info("Comparative linguistics workflow test passed")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load"""
        
        # Simulate multiple concurrent requests
        tasks = []
        test_texts = [f"Performance test {i}" for i in range(20)]
        
        async def process_text(text):
            return await self.engine.enhanced_semantic_reconstruction(
                root="test",
                contexts={"text": text, "cultural_context": "general"}
            )
        
        # Process all texts concurrently
        results = await asyncio.gather(*[process_text(text) for text in test_texts])
        
        self.assertEqual(len(results), 20)
        
        # Check performance metrics
        performance_summary = self.optimizer.get_performance_summary()
        self.assertIsInstance(performance_summary, dict)
        
        logger.info(f"Performance under load test passed: {len(results)} concurrent requests")
    
    @pytest.mark.asyncio
    async def test_cultural_boundary_enforcement(self):
        """Test cultural boundary enforcement across components"""
        
        # Test with sacred content
        sacred_text = "Quranic interpretation analysis"
        
        # Assess cultural context
        cultural_context = await self.sovereignty_manager.assess_cultural_context(
            content=sacred_text,
            metadata={"cultural_group": "islamic"}
        )
        
        # Check sovereignty compliance with insufficient permissions
        compliance = await self.sovereignty_manager.check_sovereignty_compliance(
            content_hash="boundary_test",
            cultural_context=cultural_context,
            proposed_action="interpretation",
            user_permissions={}  # No permissions
        )
        
        self.assertFalse(compliance["compliant"])
        self.assertGreater(len(compliance["violations"]), 0)
        
        # Protect the content
        protection_result = await self.sovereignty_manager.protect_cultural_content(
            content_hash="boundary_test",
            cultural_context=cultural_context,
            protection_level=CulturalSensitivityLevel.SACRED
        )
        
        self.assertTrue(protection_result["protected"])
        
        logger.info("Cultural boundary enforcement test passed")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add individual component tests
    test_suite.addTest(unittest.makeSuite(TestSirajV61Engine))
    test_suite.addTest(unittest.makeSuite(TestAdaptiveSemanticArchitecture))
    test_suite.addTest(unittest.makeSuite(TestCommunityValidationInterface))
    test_suite.addTest(unittest.makeSuite(TestCulturalSovereigntyManager))
    test_suite.addTest(unittest.makeSuite(TestMultiParadigmValidator))
    test_suite.addTest(unittest.makeSuite(TestTransformerIntegration))
    test_suite.addTest(unittest.makeSuite(TestPerformanceOptimizer))
    
    # Add integration tests
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive tests
    test_result = run_comprehensive_tests()
    
    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Tests run: {test_result.testsRun}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    
    if test_result.failures:
        print("\nFailures:")
        for test, traceback in test_result.failures:
            print(f"- {test}: {traceback}")
    
    if test_result.errors:
        print("\nErrors:")
        for test, traceback in test_result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun
    print(f"\nOverall Success Rate: {success_rate:.2%}")