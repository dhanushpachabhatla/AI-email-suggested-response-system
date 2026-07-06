"""
Evaluation Engine for AI Email Response System.

Implements multi-dimensional evaluation framework:
- Semantic Similarity (embedding-based)
- LLM-as-Judge dimensions (tone, completeness, professionalism, coherence)
- Supplementary metrics (ROUGE, BLEU)
- System-level aggregation and reporting
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.response_generator import EmbeddingManager, LLMClient, EmailPair
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Individual evaluation metric score."""
    dimension: str
    score: float              # 0.0-1.0 or 0-5 scale depending on dimension
    scale: str = "0-1"       # "0-1", "1-5"
    explanation: str = ""
    method: str = ""
    confidence: Optional[float] = None


@dataclass
class EvaluationResult:
    """Complete evaluation for a single generated response."""
    evaluation_id: str
    email_pair_id: str
    timestamp: str
    
    # Emails
    incoming_email: str
    ground_truth_response: str
    generated_response: str
    
    # Core evaluation scores
    semantic_similarity: QualityScore
    tone_appropriateness: QualityScore
    completeness: QualityScore
    professionalism: QualityScore
    coherence: QualityScore
    
    # Supplementary metrics
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    bleu_score: float = 0.0
    
    # Metadata
    retrieval_quality: float = 0.0
    generation_latency_ms: float = 0.0
    formality_level: str = ""
    email_type: str = ""
    
    # Quality assessment
    overall_quality: str = ""  # "high_quality", "acceptable", "below_threshold"
    meets_thresholds: bool = False
    flags: List[str] = field(default_factory=list)


@dataclass
class SystemPerformance:
    """Overall system evaluation metrics."""
    timestamp: str
    dataset_size: int
    
    # Per-dimension aggregates (mean, median, std)
    dimension_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality distribution
    quality_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # Failure analysis
    common_failure_modes: List[str] = field(default_factory=list)
    worst_performing_categories: List[str] = field(default_factory=list)
    
    # Latency stats
    latency_stats: Dict[str, float] = field(default_factory=dict)



# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """
    Multi-dimensional evaluation orchestrator.
    
    Evaluates generated email responses across:
    1. Semantic similarity (embedding-based)
    2. Tone appropriateness (LLM-as-judge)
    3. Completeness (LLM-as-judge)
    4. Professionalism (LLM-as-judge)
    5. Coherence (LLM-as-judge)
    6. ROUGE scores (supplementary)
    7. BLEU score (supplementary)
    """
    
    def __init__(
        self,
        embedding_mgr: EmbeddingManager,
        llm_client: LLMClient,
        thresholds: Optional[Dict] = None,
    ):
        """
        Args:
            embedding_mgr: For semantic similarity computation.
            llm_client: For LLM-as-judge evaluations.
            thresholds: Quality thresholds dict with 'high_quality' and 'acceptable'.
        """
        self.embedding_mgr = embedding_mgr
        self.llm_client = llm_client
        self.thresholds = thresholds or {
            "high_quality": {
                "semantic_similarity": 0.8,
                "llm_judge_dimensions": 4.0
            },
            "acceptable": {
                "semantic_similarity": 0.6,
                "llm_judge_dimensions": 3.0
            }
        }
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        logger.info("EvaluationEngine initialised")
    
    def evaluate_response(
        self,
        email_pair_id: str,
        incoming: str,
        generated: str,
        ground_truth: str,
        formality_level: str = "",
        email_type: str = "",
        retrieval_quality: float = 0.0,
        generation_latency_ms: float = 0.0,
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of a single response.
        
        Args:
            email_pair_id: ID of the email pair.
            incoming: The incoming email text.
            generated: Generated response text.
            ground_truth: Expected/reference response text.
            formality_level: Expected formality (formal, semi-formal, casual).
            email_type: Type of email (customer_support, professional, technical).
            retrieval_quality: Avg similarity score of retrieved examples.
            generation_latency_ms: Time taken to generate response.
            
        Returns:
            EvaluationResult with all scores and metadata.
        """
        eval_id = f"eval_{email_pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        logger.debug(f"Evaluating response for {email_pair_id}")
        
        # 1. Semantic similarity
        semantic_sim = self.compute_semantic_similarity(generated, ground_truth)
        
        # 2. LLM-as-judge dimensions
        tone_score = self.llm_judge_tone(incoming, generated, formality_level)
        completeness_score = self.llm_judge_completeness(incoming, generated)
        professionalism_score = self.llm_judge_professionalism(generated)
        coherence_score = self.llm_judge_coherence(generated)
        
        # 3. Supplementary metrics
        rouge_scores = self.compute_rouge(generated, ground_truth)
        bleu_score = self.compute_bleu(generated, ground_truth)
        
        # Quality assessment
        overall_quality, meets_thresholds, flags = self._assess_quality(
            semantic_sim.score,
            tone_score.score,
            completeness_score.score,
            professionalism_score.score,
            coherence_score.score,
        )
        
        return EvaluationResult(
            evaluation_id=eval_id,
            email_pair_id=email_pair_id,
            timestamp=timestamp,
            incoming_email=incoming,
            ground_truth_response=ground_truth,
            generated_response=generated,
            semantic_similarity=semantic_sim,
            tone_appropriateness=tone_score,
            completeness=completeness_score,
            professionalism=professionalism_score,
            coherence=coherence_score,
            rouge_scores=rouge_scores,
            bleu_score=bleu_score,
            retrieval_quality=retrieval_quality,
            generation_latency_ms=generation_latency_ms,
            formality_level=formality_level,
            email_type=email_type,
            overall_quality=overall_quality,
            meets_thresholds=meets_thresholds,
            flags=flags,
        )
    
    def compute_semantic_similarity(self, generated: str, ground_truth: str) -> QualityScore:
        """
        Compute embedding-based cosine similarity between generated and ground truth.
        
        Returns score in [0, 1] where 1 = semantically identical.
        """
        try:
            emb1 = self.embedding_mgr.embed_single(generated)
            emb2 = self.embedding_mgr.embed_single(ground_truth)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity = float(similarity)
            
            explanation = (
                f"Cosine similarity between generated and ground truth embeddings. "
                f"Score of {similarity:.3f} indicates "
                f"{'high' if similarity >= 0.8 else 'moderate' if similarity >= 0.6 else 'low'} "
                "semantic alignment."
            )
            
            return QualityScore(
                dimension="semantic_similarity",
                score=round(similarity, 4),
                scale="0-1",
                explanation=explanation,
                method="cosine_similarity_embeddings"
            )
        except Exception as e:
            logger.error(f"Semantic similarity computation failed: {e}")
            return QualityScore(
                dimension="semantic_similarity",
                score=0.0,
                scale="0-1",
                explanation=f"Error: {e}",
                method="cosine_similarity_embeddings"
            )

    
    def llm_judge_tone(self, incoming: str, generated: str, formality_level: str) -> QualityScore:
        """
        Evaluate tone appropriateness using LLM-as-judge.
        
        Scores on 1-5 scale based on whether the tone matches expected formality.
        """
        prompt = f"""Evaluate the tone appropriateness of this email response on a 1-5 scale:
- 1: Completely inappropriate tone
- 2: Poor tone match
- 3: Acceptable but could be better
- 4: Good tone match
- 5: Perfectly appropriate tone

Incoming Email: {incoming[:500]}
Expected Formality: {formality_level}
Generated Response: {generated[:500]}

Provide your evaluation in this exact format:
SCORE: [number from 1-5]
EXPLANATION: [brief explanation]
"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.3)
            score, explanation = self._parse_judge_response(response, scale=5)
            
            return QualityScore(
                dimension="tone_appropriateness",
                score=score,
                scale="1-5",
                explanation=explanation,
                method="llm_as_judge"
            )
        except Exception as e:
            logger.error(f"LLM judge (tone) failed: {e}")
            return QualityScore(
                dimension="tone_appropriateness",
                score=3.0,
                scale="1-5",
                explanation=f"Evaluation failed: {e}",
                method="llm_as_judge"
            )
    
    def llm_judge_completeness(self, incoming: str, generated: str) -> QualityScore:
        """
        Evaluate completeness using LLM-as-judge.
        
        Returns score in [0, 1] representing percentage of points addressed.
        """
        prompt = f"""Analyze whether this response addresses all points in the incoming email:

Incoming Email: {incoming[:500]}
Generated Response: {generated[:500]}

1. List all questions or points raised in the incoming email
2. For each point, mark whether the response addresses it (YES/NO)
3. Calculate completeness score as (addressed points / total points)

Provide your evaluation in this exact format:
SCORE: [decimal from 0.0 to 1.0]
EXPLANATION: [brief explanation listing which points were/weren't addressed]
"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=250, temperature=0.3)
            score, explanation = self._parse_judge_response(response, scale=1)
            
            return QualityScore(
                dimension="completeness",
                score=score,
                scale="0-1",
                explanation=explanation,
                method="llm_as_judge"
            )
        except Exception as e:
            logger.error(f"LLM judge (completeness) failed: {e}")
            return QualityScore(
                dimension="completeness",
                score=0.5,
                scale="0-1",
                explanation=f"Evaluation failed: {e}",
                method="llm_as_judge"
            )
    
    def llm_judge_professionalism(self, generated: str) -> QualityScore:
        """
        Evaluate professionalism using LLM-as-judge.
        
        Scores on 1-5 scale based on grammar, spelling, structure, courtesy.
        """
        prompt = f"""Evaluate the professionalism of this email response (1-5 scale):
- Grammar and spelling correctness
- Clear sentence structure
- Professional courtesy and tone
- Appropriate greeting and closing

Response: {generated[:500]}

Provide your evaluation in this exact format:
SCORE: [number from 1-5]
EXPLANATION: [brief explanation noting any issues]
"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.3)
            score, explanation = self._parse_judge_response(response, scale=5)
            
            return QualityScore(
                dimension="professionalism",
                score=score,
                scale="1-5",
                explanation=explanation,
                method="llm_as_judge"
            )
        except Exception as e:
            logger.error(f"LLM judge (professionalism) failed: {e}")
            return QualityScore(
                dimension="professionalism",
                score=3.0,
                scale="1-5",
                explanation=f"Evaluation failed: {e}",
                method="llm_as_judge"
            )
    
    def llm_judge_coherence(self, generated: str) -> QualityScore:
        """
        Evaluate coherence using LLM-as-judge.
        
        Scores on 1-5 scale based on logical flow, consistency, readability.
        """
        prompt = f"""Evaluate the coherence of this email response (1-5 scale):
- Logical flow between ideas
- Internal consistency (no contradictions)
- Clear and easy to follow
- Well-structured paragraphs

Response: {generated[:500]}

Provide your evaluation in this exact format:
SCORE: [number from 1-5]
EXPLANATION: [brief explanation noting any coherence issues]
"""
        
        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.3)
            score, explanation = self._parse_judge_response(response, scale=5)
            
            return QualityScore(
                dimension="coherence",
                score=score,
                scale="1-5",
                explanation=explanation,
                method="llm_as_judge"
            )
        except Exception as e:
            logger.error(f"LLM judge (coherence) failed: {e}")
            return QualityScore(
                dimension="coherence",
                score=3.0,
                scale="1-5",
                explanation=f"Evaluation failed: {e}",
                method="llm_as_judge"
            )
    
    def _parse_judge_response(self, response: str, scale: float) -> tuple[float, str]:
        """
        Parse LLM judge response to extract score and explanation.
        
        Args:
            response: Raw LLM output.
            scale: Maximum value of the scale (1 or 5).
            
        Returns:
            Tuple of (score, explanation).
        """
        score_pattern = r"SCORE:\s*([0-9.]+)"
        explanation_pattern = r"EXPLANATION:\s*(.+?)(?:\n\n|$)"
        
        score_match = re.search(score_pattern, response, re.IGNORECASE)
        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if score_match:
            try:
                score = float(score_match.group(1))
                # Clamp to valid range
                score = max(0, min(scale, score))
            except ValueError:
                score = scale / 2  # Default to middle of scale
        else:
            logger.warning(f"Could not parse score from LLM judge response: {response[:100]}")
            score = scale / 2
        
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        
        return score, explanation

    
    def compute_rouge(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Note: ROUGE has known limitations for email evaluation (paraphrasing blindness,
        reference dependence). Used only as supplementary indicators.
        """
        try:
            scores = self.rouge_scorer.score(ground_truth, generated)
            return {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
            }
        except Exception as e:
            logger.error(f"ROUGE computation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def compute_bleu(self, generated: str, ground_truth: str) -> float:
        """
        Compute BLEU score.
        
        Note: BLEU has known limitations for email evaluation (n-gram matching,
        single reference). Used only as supplementary indicator.
        """
        try:
            # Tokenize
            reference = [ground_truth.split()]
            hypothesis = generated.split()
            
            # Use smoothing to avoid zero scores
            smoothing = SmoothingFunction().method1
            score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
            
            return round(score, 4)
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return 0.0
    
    def _assess_quality(
        self,
        semantic_sim: float,
        tone: float,
        completeness: float,
        professionalism: float,
        coherence: float,
    ) -> tuple[str, bool, List[str]]:
        """
        Assess overall quality and identify flags.
        
        Returns:
            Tuple of (overall_quality_level, meets_thresholds, list_of_flags)
        """
        flags = []
        
        # Convert 1-5 scores to 0-1 scale for comparison
        tone_normalized = tone / 5.0
        professionalism_normalized = professionalism / 5.0
        coherence_normalized = coherence / 5.0
        
        # Check thresholds
        high_sem = semantic_sim >= self.thresholds["high_quality"]["semantic_similarity"]
        high_judge = (
            tone >= self.thresholds["high_quality"]["llm_judge_dimensions"]
            and completeness >= (self.thresholds["high_quality"]["llm_judge_dimensions"] / 5.0)
            and professionalism >= self.thresholds["high_quality"]["llm_judge_dimensions"]
            and coherence >= self.thresholds["high_quality"]["llm_judge_dimensions"]
        )
        
        accept_sem = semantic_sim >= self.thresholds["acceptable"]["semantic_similarity"]
        accept_judge = (
            tone >= self.thresholds["acceptable"]["llm_judge_dimensions"]
            and completeness >= (self.thresholds["acceptable"]["llm_judge_dimensions"] / 5.0)
            and professionalism >= self.thresholds["acceptable"]["llm_judge_dimensions"]
            and coherence >= self.thresholds["acceptable"]["llm_judge_dimensions"]
        )
        
        # Determine overall quality
        if high_sem and high_judge:
            overall = "high_quality"
            meets_thresholds = True
        elif accept_sem and accept_judge:
            overall = "acceptable"
            meets_thresholds = True
        else:
            overall = "below_threshold"
            meets_thresholds = False
        
        # Identify specific flags
        if semantic_sim < self.thresholds["acceptable"]["semantic_similarity"]:
            flags.append("low_semantic_similarity")
        if tone < self.thresholds["acceptable"]["llm_judge_dimensions"]:
            flags.append("inappropriate_tone")
        if completeness < (self.thresholds["acceptable"]["llm_judge_dimensions"] / 5.0):
            flags.append("incomplete_response")
        if professionalism < self.thresholds["acceptable"]["llm_judge_dimensions"]:
            flags.append("unprofessional")
        if coherence < self.thresholds["acceptable"]["llm_judge_dimensions"]:
            flags.append("incoherent")
        
        return overall, meets_thresholds, flags


# ---------------------------------------------------------------------------
# Aggregation Engine
# ---------------------------------------------------------------------------

class AggregationEngine:
    """
    Aggregates per-response evaluation results into system-level metrics.
    """
    
    def __init__(self):
        logger.info("AggregationEngine initialised")
    
    def aggregate_results(self, results: List[EvaluationResult]) -> SystemPerformance:
        """
        Compute overall system performance from individual evaluation results.
        
        Args:
            results: List of EvaluationResult objects.
            
        Returns:
            SystemPerformance with aggregate statistics.
        """
        if not results:
            logger.warning("No evaluation results to aggregate")
            return SystemPerformance(
                timestamp=datetime.now().isoformat(),
                dataset_size=0
            )
        
        logger.info(f"Aggregating {len(results)} evaluation results")
        
        # Extract scores
        semantic_sims = [r.semantic_similarity.score for r in results]
        tones = [r.tone_appropriateness.score for r in results]
        completenesses = [r.completeness.score for r in results]
        professionalisms = [r.professionalism.score for r in results]
        coherences = [r.coherence.score for r in results]
        latencies = [r.generation_latency_ms for r in results]
        
        # Compute dimension statistics
        dimension_stats = {
            "semantic_similarity": self._compute_stats(semantic_sims),
            "tone_appropriateness": self._compute_stats(tones),
            "completeness": self._compute_stats(completenesses),
            "professionalism": self._compute_stats(professionalisms),
            "coherence": self._compute_stats(coherences),
        }
        
        # Compute quality distribution
        quality_counts = defaultdict(int)
        for r in results:
            quality_counts[r.overall_quality] += 1
        
        quality_distribution = {
            "high_quality": {
                "count": quality_counts["high_quality"],
                "percentage": round(quality_counts["high_quality"] / len(results) * 100, 2)
            },
            "acceptable": {
                "count": quality_counts["acceptable"],
                "percentage": round(quality_counts["acceptable"] / len(results) * 100, 2)
            },
            "below_threshold": {
                "count": quality_counts["below_threshold"],
                "percentage": round(quality_counts["below_threshold"] / len(results) * 100, 2)
            }
        }
        
        # Latency statistics
        latency_stats = self._compute_stats(latencies)
        
        # Failure analysis
        common_failures = self._analyze_failures(results)
        worst_categories = self._identify_worst_categories(results)
        
        return SystemPerformance(
            timestamp=datetime.now().isoformat(),
            dataset_size=len(results),
            dimension_stats=dimension_stats,
            quality_distribution=quality_distribution,
            common_failure_modes=common_failures,
            worst_performing_categories=worst_categories,
            latency_stats=latency_stats,
        )
    
    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute mean, median, std, min, max for a list of values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_val = variance ** 0.5
        
        return {
            "mean": round(mean_val, 4),
            "median": round(median, 4),
            "std": round(std_val, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    
    def _analyze_failures(self, results: List[EvaluationResult]) -> List[str]:
        """Identify most common failure modes."""
        flag_counts = defaultdict(int)
        for r in results:
            for flag in r.flags:
                flag_counts[flag] += 1
        
        # Sort by frequency
        sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{flag} ({count} occurrences)" for flag, count in sorted_flags[:5]]
    
    def _identify_worst_categories(self, results: List[EvaluationResult]) -> List[str]:
        """Identify categories with worst average performance."""
        category_scores = defaultdict(list)
        
        for r in results:
            if r.email_type:
                # Average of all dimension scores
                avg_score = (
                    r.semantic_similarity.score +
                    r.tone_appropriateness.score / 5.0 +
                    r.completeness.score +
                    r.professionalism.score / 5.0 +
                    r.coherence.score / 5.0
                ) / 5.0
                category_scores[r.email_type].append(avg_score)
        
        # Compute average score per category
        category_avgs = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }
        
        # Sort by lowest performance
        sorted_cats = sorted(category_avgs.items(), key=lambda x: x[1])
        return [f"{cat} (avg score: {score:.3f})" for cat, score in sorted_cats]
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        performance: SystemPerformance,
        output_path: str,
    ) -> None:
        """
        Generate comprehensive evaluation report as JSON.
        
        Args:
            results: List of individual EvaluationResult objects.
            performance: SystemPerformance aggregate statistics.
            output_path: Path to save the report JSON file.
        """
        logger.info(f"Generating evaluation report: {output_path}")
        
        report = {
            "evaluation_summary": asdict(performance),
            "per_response_results": [
                {
                    "evaluation_id": r.evaluation_id,
                    "email_pair_id": r.email_pair_id,
                    "scores": {
                        "semantic_similarity": asdict(r.semantic_similarity),
                        "tone_appropriateness": asdict(r.tone_appropriateness),
                        "completeness": asdict(r.completeness),
                        "professionalism": asdict(r.professionalism),
                        "coherence": asdict(r.coherence),
                    },
                    "supplementary_metrics": {
                        "rouge_scores": r.rouge_scores,
                        "bleu_score": r.bleu_score,
                    },
                    "metadata": {
                        "retrieval_quality": r.retrieval_quality,
                        "generation_latency_ms": r.generation_latency_ms,
                        "formality_level": r.formality_level,
                        "email_type": r.email_type,
                    },
                    "quality_assessment": {
                        "overall": r.overall_quality,
                        "meets_thresholds": r.meets_thresholds,
                        "flags": r.flags,
                    },
                    "emails": {
                        "incoming": r.incoming_email[:200] + "...",
                        "ground_truth": r.ground_truth_response[:200] + "...",
                        "generated": r.generated_response[:200] + "...",
                    },
                }
                for r in results
            ],
        }
        
        # Save to file
        from pathlib import Path
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved: {output_path}")
        
        # Also generate a human-readable summary
        summary_path = str(output_file).replace('.json', '_summary.txt')
        self._generate_text_summary(performance, summary_path)
    
    def _generate_text_summary(self, performance: SystemPerformance, output_path: str) -> None:
        """Generate human-readable text summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("AI EMAIL RESPONSE SYSTEM - EVALUATION SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {performance.timestamp}")
        lines.append(f"Dataset Size: {performance.dataset_size} responses evaluated")
        lines.append("")
        
        lines.append("DIMENSION STATISTICS (Mean ± Std)")
        lines.append("-" * 70)
        for dim, stats in performance.dimension_stats.items():
            lines.append(f"  {dim:25s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                        f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
        lines.append("")
        
        lines.append("QUALITY DISTRIBUTION")
        lines.append("-" * 70)
        for level, data in performance.quality_distribution.items():
            lines.append(f"  {level:20s}: {data['count']:3d} ({data['percentage']:5.1f}%)")
        lines.append("")
        
        if performance.common_failure_modes:
            lines.append("COMMON FAILURE MODES")
            lines.append("-" * 70)
            for failure in performance.common_failure_modes:
                lines.append(f"  - {failure}")
            lines.append("")
        
        if performance.worst_performing_categories:
            lines.append("WORST PERFORMING CATEGORIES")
            lines.append("-" * 70)
            for cat in performance.worst_performing_categories:
                lines.append(f"  - {cat}")
            lines.append("")
        
        lines.append("LATENCY STATISTICS (milliseconds)")
        lines.append("-" * 70)
        lat = performance.latency_stats
        lines.append(f"  Mean: {lat['mean']:.2f}ms")
        lines.append(f"  Median: {lat['median']:.2f}ms")
        lines.append(f"  Min: {lat['min']:.2f}ms, Max: {lat['max']:.2f}ms")
        lines.append("")
        lines.append("=" * 70)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Text summary saved: {output_path}")
