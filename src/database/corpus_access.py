"""
Corpus Data Access Layer for SIRAJ v6.1
Provides access to Quranic, Hadith, and Classical Arabic text corpora
Implements real data retrieval to replace stub methods in the engine
"""

import asyncio
import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter

from src.database.connection_manager import ConnectionManager
from src.utils.exceptions import SirajProcessingError
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QuranicOccurrence:
    """Represents an occurrence of a root in the Quran"""
    surah: int
    verse: int
    word_position: int
    word_form: str
    arabic_text: str
    transliteration: Optional[str] = None
    translation: Optional[str] = None
    context: Optional[str] = None
    linguistic_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HadithOccurrence:
    """Represents an occurrence of a root in Hadith literature"""
    collection: str
    book_number: int
    hadith_number: int
    word_form: str
    arabic_text: str
    english_text: Optional[str] = None
    narrator_chain: Optional[str] = None
    authenticity_grade: Optional[str] = None
    subject_tags: List[str] = field(default_factory=list)
    context: Optional[str] = None


@dataclass
class ClassicalTextOccurrence:
    """Represents an occurrence of a root in classical Arabic literature"""
    work_title: str
    author: str
    period: str
    genre: str
    word_form: str
    text_excerpt: str
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    linguistic_features: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class RootAnalysisResult:
    """Comprehensive analysis results for a root"""
    root: str
    total_occurrences: int
    quranic_occurrences: List[QuranicOccurrence]
    hadith_occurrences: List[HadithOccurrence] 
    classical_occurrences: List[ClassicalTextOccurrence]
    semantic_fields: List[str]
    word_forms: List[str]
    confidence_scores: Dict[str, float]
    statistical_analysis: Dict[str, Any]


class CorpusDataAccess:
    """
    Corpus Data Access Layer providing real text retrieval and analysis
    
    Replaces stub implementations in SIRAJ engine with actual database queries
    and linguistic analysis of Arabic text corpora.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn_mgr = connection_manager
        self.arabic_root_pattern = re.compile(r'[\u0621-\u063A\u0641-\u064A]+')  # Arabic letters
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # Arabic diacritics
        
        # Semantic field mappings (expandable)
        self.semantic_fields = {
            'knowledge': ['علم', 'عرف', 'درس', 'كتب', 'قرأ'],
            'worship': ['صلى', 'صوم', 'حج', 'دعا', 'سبح'],
            'social': ['أهل', 'قوم', 'ناس', 'مجتمع', 'عائل'],
            'nature': ['أرض', 'سما', 'شمس', 'قمر', 'نجم'],
            'time': ['وقت', 'يوم', 'ليل', 'صبح', 'عصر']
        }
    
    async def analyze_quranic_usage(self, root: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of Quranic usage for a given root
        Replaces stub implementation in siraj_v6_1_engine.py:893
        """
        try:
            logger.info(f"Analyzing Quranic usage for root: {root}")
            
            # Normalize root (remove diacritics and extra characters)
            normalized_root = self._normalize_arabic_text(root)
            
            async with self.conn_mgr.get_corpus_connection() as conn:
                # Query for verses containing the root
                query = """
                SELECT surah_number, verse_number, arabic_text, transliteration, 
                       translation_en, revelation_context, linguistic_features, root_analysis
                FROM quranic_verses 
                WHERE root_analysis ? $1 OR arabic_text ~ $2
                ORDER BY surah_number, verse_number
                """
                
                # Create regex pattern for root variants
                root_pattern = self._create_root_pattern(normalized_root)
                
                rows = await conn.fetch(query, normalized_root, root_pattern)
                
                if not rows:
                    # If no direct matches, try broader search
                    rows = await self._fallback_quranic_search(conn, normalized_root)
                
                occurrences = []
                semantic_fields = set()
                word_forms = set()
                
                for row in rows:
                    # Extract root occurrences from the verse
                    verse_occurrences = self._extract_root_from_verse(
                        row['arabic_text'], normalized_root, row
                    )
                    occurrences.extend(verse_occurrences)
                    
                    # Collect semantic fields
                    if row['root_analysis'] and isinstance(row['root_analysis'], dict):
                        fields = row['root_analysis'].get('semantic_fields', [])
                        semantic_fields.update(fields)
                    
                    # Collect word forms
                    forms = self._extract_word_forms(row['arabic_text'], normalized_root)
                    word_forms.update(forms)
                
                # Calculate frequency statistics
                frequency_analysis = self._calculate_frequency_statistics(occurrences)
                
                # Determine semantic fields if not found in metadata
                if not semantic_fields:
                    semantic_fields = self._infer_semantic_fields(occurrences)
                
                # Calculate confidence based on occurrence count and text quality
                confidence = self._calculate_quranic_confidence(occurrences, len(rows))
                
                return {
                    "occurrences": len(occurrences),
                    "verses_found": len(rows),
                    "contexts": [occ.arabic_text[:100] + "..." for occ in occurrences[:5]],
                    "semantic_fields": list(semantic_fields),
                    "word_forms": list(word_forms)[:10],  # Limit to top 10 forms
                    "frequency_analysis": frequency_analysis,
                    "confidence": confidence,
                    "detailed_occurrences": [
                        {
                            "surah": occ.surah,
                            "verse": occ.verse, 
                            "word_form": occ.word_form,
                            "context": occ.arabic_text[:200]
                        } for occ in occurrences[:10]
                    ]
                }
        
        except Exception as e:
            logger.error(f"Error analyzing Quranic usage for root '{root}': {e}")
            return {"error": str(e)}
    
    async def analyze_hadith_references(self, root: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of Hadith references for a given root
        Replaces stub implementation in siraj_v6_1_engine.py:939
        """
        try:
            logger.info(f"Analyzing Hadith references for root: {root}")
            
            normalized_root = self._normalize_arabic_text(root)
            
            async with self.conn_mgr.get_corpus_connection() as conn:
                query = """
                SELECT collection_name, book_number, hadith_number, arabic_text,
                       english_text, narrator_chain, authenticity_grade, subject_tags,
                       linguistic_analysis, root_occurrences
                FROM hadith_collection 
                WHERE root_occurrences ? $1 OR arabic_text ~ $2
                ORDER BY collection_name, book_number, hadith_number
                LIMIT 100
                """
                
                root_pattern = self._create_root_pattern(normalized_root)
                rows = await conn.fetch(query, normalized_root, root_pattern)
                
                if not rows:
                    rows = await self._fallback_hadith_search(conn, normalized_root)
                
                occurrences = []
                collections = set()
                authenticity_grades = defaultdict(int)
                subject_analysis = defaultdict(int)
                
                for row in rows:
                    occurrence = HadithOccurrence(
                        collection=row['collection_name'],
                        book_number=row['book_number'] or 0,
                        hadith_number=row['hadith_number'] or 0,
                        word_form=self._extract_root_form(row['arabic_text'], normalized_root),
                        arabic_text=row['arabic_text'],
                        english_text=row['english_text'],
                        narrator_chain=row['narrator_chain'],
                        authenticity_grade=row['authenticity_grade'],
                        subject_tags=row['subject_tags'] or []
                    )
                    occurrences.append(occurrence)
                    collections.add(row['collection_name'])
                    
                    if row['authenticity_grade']:
                        authenticity_grades[row['authenticity_grade']] += 1
                    
                    for tag in (row['subject_tags'] or []):
                        subject_analysis[tag] += 1
                
                # Statistical analysis
                collection_distribution = {col: sum(1 for occ in occurrences if occ.collection == col) 
                                        for col in collections}
                
                confidence = self._calculate_hadith_confidence(occurrences, authenticity_grades)
                
                return {
                    "total_references": len(occurrences),
                    "collections": list(collections),
                    "collection_distribution": collection_distribution,
                    "authenticity_analysis": dict(authenticity_grades),
                    "subject_analysis": dict(sorted(subject_analysis.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "semantic_analysis": self._analyze_hadith_semantics(occurrences),
                    "confidence": confidence,
                    "sample_references": [
                        {
                            "collection": occ.collection,
                            "reference": f"{occ.book_number}:{occ.hadith_number}",
                            "word_form": occ.word_form,
                            "authenticity": occ.authenticity_grade,
                            "excerpt": occ.arabic_text[:150] + "..." if len(occ.arabic_text) > 150 else occ.arabic_text
                        } for occ in occurrences[:5]
                    ]
                }
        
        except Exception as e:
            logger.error(f"Error analyzing Hadith references for root '{root}': {e}")
            return {"error": str(e)}
    
    async def analyze_classical_arabic_usage(self, root: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of classical Arabic literature usage
        Replaces stub implementation in siraj_v6_1_engine.py:981
        """
        try:
            logger.info(f"Analyzing classical Arabic usage for root: {root}")
            
            normalized_root = self._normalize_arabic_text(root)
            
            async with self.conn_mgr.get_corpus_connection() as conn:
                query = """
                SELECT work_title, author_name, period, genre, text_excerpt,
                       linguistic_features, cultural_context, root_analysis
                FROM classical_texts
                WHERE root_analysis ? $1 OR text_excerpt ~ $2
                ORDER BY period, author_name
                LIMIT 50
                """
                
                root_pattern = self._create_root_pattern(normalized_root)
                rows = await conn.fetch(query, normalized_root, root_pattern)
                
                if not rows:
                    rows = await self._fallback_classical_search(conn, normalized_root)
                
                occurrences = []
                periods = defaultdict(int)
                genres = defaultdict(int)
                authors = set()
                
                for row in rows:
                    occurrence = ClassicalTextOccurrence(
                        work_title=row['work_title'],
                        author=row['author_name'],
                        period=row['period'] or 'unknown',
                        genre=row['genre'] or 'unknown',
                        word_form=self._extract_root_form(row['text_excerpt'], normalized_root),
                        text_excerpt=row['text_excerpt'],
                        cultural_context=row['cultural_context'] or {},
                        linguistic_features=row['linguistic_features'] or {}
                    )
                    occurrences.append(occurrence)
                    periods[occurrence.period] += 1
                    genres[occurrence.genre] += 1
                    authors.add(occurrence.author)
                
                # Analyze historical development
                historical_development = self._analyze_historical_development(occurrences)
                
                # Analyze genre distribution
                genre_analysis = dict(sorted(genres.items(), key=lambda x: x[1], reverse=True))
                
                # Calculate confidence based on source diversity and occurrence count
                confidence = self._calculate_classical_confidence(occurrences, len(authors), len(periods))
                
                return {
                    "total_occurrences": len(occurrences),
                    "unique_authors": len(authors),
                    "period_distribution": dict(periods),
                    "genre_analysis": genre_analysis,
                    "historical_development": historical_development,
                    "literary_usage": [
                        {
                            "work": occ.work_title,
                            "author": occ.author,
                            "period": occ.period,
                            "genre": occ.genre,
                            "excerpt": occ.text_excerpt[:200] + "..." if len(occ.text_excerpt) > 200 else occ.text_excerpt
                        } for occ in occurrences[:5]
                    ],
                    "grammatical_patterns": self._analyze_grammatical_patterns(occurrences),
                    "semantic_evolution": self._analyze_semantic_evolution(occurrences),
                    "confidence": confidence
                }
        
        except Exception as e:
            logger.error(f"Error analyzing classical Arabic usage for root '{root}': {e}")
            return {"error": str(e)}
    
    # Helper methods
    
    def _normalize_arabic_text(self, text: str) -> str:
        """Remove diacritics and normalize Arabic text"""
        # Remove diacritics
        normalized = self.diacritics_pattern.sub('', text)
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized.strip()
    
    def _create_root_pattern(self, root: str) -> str:
        """Create regex pattern for root matching with variations"""
        # Simple pattern - could be enhanced with morphological rules
        letters = list(root)
        if len(letters) >= 3:
            # Pattern allowing for inflections between root letters
            pattern = f".*{letters[0]}.*{letters[1]}.*{letters[2]}.*"
            return pattern
        return f".*{root}.*"
    
    def _extract_root_from_verse(self, verse_text: str, root: str, verse_data: dict) -> List[QuranicOccurrence]:
        """Extract occurrences of root from verse text"""
        occurrences = []
        words = verse_text.split()
        
        for i, word in enumerate(words):
            if self._word_contains_root(word, root):
                occurrence = QuranicOccurrence(
                    surah=verse_data['surah_number'],
                    verse=verse_data['verse_number'],
                    word_position=i,
                    word_form=word,
                    arabic_text=verse_text,
                    transliteration=verse_data.get('transliteration'),
                    translation=verse_data.get('translation_en'),
                    context=verse_data.get('revelation_context'),
                    linguistic_features=verse_data.get('linguistic_features', {})
                )
                occurrences.append(occurrence)
        
        return occurrences
    
    def _word_contains_root(self, word: str, root: str) -> bool:
        """Check if word contains the given root"""
        clean_word = self._normalize_arabic_text(word)
        clean_root = self._normalize_arabic_text(root)
        
        # Simple containment check - could be enhanced with morphological analysis
        return clean_root in clean_word or self._fuzzy_root_match(clean_word, clean_root)
    
    def _fuzzy_root_match(self, word: str, root: str) -> bool:
        """Fuzzy matching for root identification"""
        if len(root) < 3:
            return False
        
        # Check if root letters appear in order (allowing for insertions)
        root_chars = list(root)
        word_chars = list(word)
        
        i = 0  # root index
        for char in word_chars:
            if i < len(root_chars) and char == root_chars[i]:
                i += 1
        
        # Return True if we found all root characters in order
        return i == len(root_chars)
    
    def _extract_word_forms(self, text: str, root: str) -> Set[str]:
        """Extract different word forms containing the root"""
        words = text.split()
        forms = set()
        
        for word in words:
            if self._word_contains_root(word, root):
                forms.add(self._normalize_arabic_text(word))
        
        return forms
    
    def _calculate_frequency_statistics(self, occurrences: List[QuranicOccurrence]) -> Dict[str, Any]:
        """Calculate frequency statistics for Quranic occurrences"""
        if not occurrences:
            return {}
        
        surah_counts = defaultdict(int)
        word_form_counts = defaultdict(int)
        
        for occ in occurrences:
            surah_counts[occ.surah] += 1
            word_form_counts[occ.word_form] += 1
        
        return {
            "total_occurrences": len(occurrences),
            "unique_surahs": len(surah_counts),
            "unique_word_forms": len(word_form_counts),
            "most_frequent_surahs": dict(sorted(surah_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_frequent_forms": dict(sorted(word_form_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _infer_semantic_fields(self, occurrences: List[QuranicOccurrence]) -> Set[str]:
        """Infer semantic fields based on context analysis"""
        fields = set()
        
        for field_name, keywords in self.semantic_fields.items():
            for occ in occurrences:
                text = occ.arabic_text.lower()
                if any(keyword in text for keyword in keywords):
                    fields.add(field_name)
        
        return fields
    
    def _calculate_quranic_confidence(self, occurrences: List[QuranicOccurrence], verse_count: int) -> float:
        """Calculate confidence score for Quranic analysis"""
        if not occurrences:
            return 0.0
        
        # Base confidence on occurrence count
        occurrence_factor = min(len(occurrences) / 10, 1.0)  # Max at 10 occurrences
        verse_factor = min(verse_count / 5, 1.0)  # Max at 5 verses
        
        # Quality factors
        has_translation = sum(1 for occ in occurrences if occ.translation) / len(occurrences)
        has_context = sum(1 for occ in occurrences if occ.context) / len(occurrences)
        
        confidence = (occurrence_factor * 0.4 + verse_factor * 0.3 + 
                     has_translation * 0.15 + has_context * 0.15)
        
        return min(confidence, 1.0)
    
    def _extract_root_form(self, text: str, root: str) -> str:
        """Extract the actual word form containing the root"""
        words = text.split()
        for word in words:
            if self._word_contains_root(word, root):
                return self._normalize_arabic_text(word)
        return root  # Fallback
    
    def _analyze_hadith_semantics(self, occurrences: List[HadithOccurrence]) -> Dict[str, Any]:
        """Analyze semantic patterns in Hadith occurrences"""
        themes = defaultdict(int)
        contexts = []
        
        for occ in occurrences:
            # Analyze subject tags for themes
            for tag in occ.subject_tags:
                themes[tag] += 1
            
            # Collect context samples
            if occ.english_text:
                contexts.append(occ.english_text[:100])
        
        return {
            "dominant_themes": dict(sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]),
            "context_variety": len(set(contexts)),
            "sample_contexts": contexts[:3]
        }
    
    def _calculate_hadith_confidence(self, occurrences: List[HadithOccurrence], 
                                   authenticity_grades: Dict[str, int]) -> float:
        """Calculate confidence score for Hadith analysis"""
        if not occurrences:
            return 0.0
        
        # Base confidence on occurrence count
        occurrence_factor = min(len(occurrences) / 20, 1.0)  # Max at 20 occurrences
        
        # Authenticity factor (higher for sahih, lower for weak)
        authenticity_weights = {
            'sahih': 1.0,
            'hasan': 0.8,
            'daif': 0.3,
            'mawdu': 0.1
        }
        
        total_weight = 0
        for grade, count in authenticity_grades.items():
            weight = authenticity_weights.get(grade.lower(), 0.5)
            total_weight += weight * count
        
        authenticity_factor = total_weight / len(occurrences) if occurrences else 0
        
        # Collection diversity factor
        collections = set(occ.collection for occ in occurrences)
        diversity_factor = min(len(collections) / 6, 1.0)  # Max at 6 collections
        
        confidence = (occurrence_factor * 0.5 + authenticity_factor * 0.3 + diversity_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def _analyze_historical_development(self, occurrences: List[ClassicalTextOccurrence]) -> Dict[str, Any]:
        """Analyze historical development of usage"""
        period_order = ['pre-islamic', 'early_islamic', 'umayyad', 'abbasid', 'medieval', 'modern']
        period_usage = defaultdict(list)
        
        for occ in occurrences:
            period_key = occ.period.lower().replace(' ', '_')
            period_usage[period_key].append(occ)
        
        development = {}
        for period in period_order:
            if period in period_usage:
                usage = period_usage[period]
                development[period] = {
                    "count": len(usage),
                    "authors": list(set(occ.author for occ in usage)),
                    "genres": list(set(occ.genre for occ in usage))
                }
        
        return development
    
    def _analyze_grammatical_patterns(self, occurrences: List[ClassicalTextOccurrence]) -> Dict[str, Any]:
        """Analyze grammatical usage patterns"""
        patterns = defaultdict(int)
        
        for occ in occurrences:
            # Simple pattern detection (could be enhanced with NLP)
            word_form = occ.word_form
            if word_form.startswith('ال'):  # Definite article
                patterns['definite'] += 1
            if word_form.endswith('ة'):  # Feminine marker
                patterns['feminine'] += 1
            if len(word_form) > 5:  # Long forms (likely derived)
                patterns['derived'] += 1
        
        return dict(patterns)
    
    def _analyze_semantic_evolution(self, occurrences: List[ClassicalTextOccurrence]) -> Dict[str, Any]:
        """Analyze semantic evolution over periods"""
        # Placeholder for semantic evolution analysis
        # Would require more sophisticated NLP and historical linguistics
        return {
            "stability_score": 0.8,  # How stable the meaning has been
            "evolution_direction": "expansion",  # expansion, narrowing, shift
            "modern_relevance": 0.9
        }
    
    def _calculate_classical_confidence(self, occurrences: List[ClassicalTextOccurrence], 
                                      author_count: int, period_count: int) -> float:
        """Calculate confidence score for classical Arabic analysis"""
        if not occurrences:
            return 0.0
        
        occurrence_factor = min(len(occurrences) / 15, 1.0)  # Max at 15 occurrences
        author_diversity = min(author_count / 10, 1.0)  # Max at 10 authors
        period_diversity = min(period_count / 5, 1.0)  # Max at 5 periods
        
        confidence = (occurrence_factor * 0.5 + author_diversity * 0.3 + period_diversity * 0.2)
        
        return min(confidence, 1.0)
    
    # Fallback search methods for when direct queries don't find results
    
    async def _fallback_quranic_search(self, conn, root: str) -> List:
        """Fallback search when direct root search fails"""
        # Search by individual letters of the root
        if len(root) >= 3:
            letters = list(root)
            query = """
            SELECT surah_number, verse_number, arabic_text, transliteration,
                   translation_en, revelation_context, linguistic_features, root_analysis
            FROM quranic_verses 
            WHERE arabic_text LIKE $1 AND arabic_text LIKE $2 AND arabic_text LIKE $3
            LIMIT 20
            """
            return await conn.fetch(query, f"%{letters[0]}%", f"%{letters[1]}%", f"%{letters[2]}%")
        return []
    
    async def _fallback_hadith_search(self, conn, root: str) -> List:
        """Fallback search for Hadith when direct search fails"""
        if len(root) >= 3:
            letters = list(root)
            query = """
            SELECT collection_name, book_number, hadith_number, arabic_text,
                   english_text, narrator_chain, authenticity_grade, subject_tags,
                   linguistic_analysis, root_occurrences
            FROM hadith_collection 
            WHERE arabic_text LIKE $1 AND arabic_text LIKE $2 AND arabic_text LIKE $3
            LIMIT 30
            """
            return await conn.fetch(query, f"%{letters[0]}%", f"%{letters[1]}%", f"%{letters[2]}%")
        return []
    
    async def _fallback_classical_search(self, conn, root: str) -> List:
        """Fallback search for classical texts when direct search fails"""
        if len(root) >= 3:
            letters = list(root)
            query = """
            SELECT work_title, author_name, period, genre, text_excerpt,
                   linguistic_features, cultural_context, root_analysis
            FROM classical_texts
            WHERE text_excerpt LIKE $1 AND text_excerpt LIKE $2 AND text_excerpt LIKE $3
            LIMIT 25
            """
            return await conn.fetch(query, f"%{letters[0]}%", f"%{letters[1]}%", f"%{letters[2]}%")
        return []