"""
AI Email Response System - Main CLI Pipeline.

Usage:
    python main.py                           # Run full evaluation
    python main.py --generate-dataset        # Regenerate dataset first
    python main.py --test-size 10            # Quick test with 10 emails
    python main.py --config my_config.yaml   # Custom config
    python main.py --output-dir my_results   # Custom output directory
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Email Response System - Evaluate LLM-generated email responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--generate-dataset", action="store_true",
        help="Regenerate synthetic dataset even if it already exists"
    )
    parser.add_argument(
        "--test-size", type=int, default=None,
        help="Process only first N emails (useful for quick testing)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip LLM response generation (for testing evaluation only)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # -----------------------------------------------------------------------
    # 1. Load configuration
    # -----------------------------------------------------------------------
    from src.config import Config
    from src.utils import get_logger
    
    print(f"[1/6] Loading configuration from '{args.config}'...")
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    logger = get_logger("main")
    logger.info("Configuration loaded successfully")
    
    output_dir = args.output_dir or config.get("output.output_dir", "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------------------------
    # 2. Load or generate dataset
    # -----------------------------------------------------------------------
    from src.dataset_manager import DatasetManager
    
    print("[2/6] Loading dataset...")
    manager = DatasetManager()
    dataset_path = config.get("dataset.path", "data/email_dataset.json")
    
    if args.generate_dataset or not Path(dataset_path).exists():
        print(f"  Generating synthetic dataset ({config.get('dataset.synthetic_size', 100)} pairs)...")
        pairs = manager.generate_synthetic_dataset(
            size=config.get("dataset.synthetic_size", 100),
            filepath=dataset_path
        )
        print(f"  Dataset generated and saved to '{dataset_path}'")
    else:
        pairs = manager.load_dataset(dataset_path)
        print(f"  Loaded {len(pairs)} email pairs from '{dataset_path}'")
    
    if args.test_size:
        pairs = pairs[:args.test_size]
        print(f"  Using subset: {len(pairs)} pairs (--test-size {args.test_size})")
    
    if len(pairs) < config.get("dataset.min_pairs", 50) and not args.test_size:
        print(f"WARNING: Dataset has only {len(pairs)} pairs (minimum: {config.get('dataset.min_pairs', 50)})")
    
    # -----------------------------------------------------------------------
    # 3. Initialise components
    # -----------------------------------------------------------------------
    from src.response_generator import EmbeddingManager, VectorStore, LLMClient, ResponseGenerator
    from src.evaluation_engine import EvaluationEngine, AggregationEngine
    
    print("[3/6] Initialising components...")
    
    # Embedding model
    print(f"  Loading embedding model '{config.get('embeddings.model')}'...")
    emb_mgr = EmbeddingManager(model_name=config.get("embeddings.model", "all-MiniLM-L6-v2"))
    
    # Vector store
    vector_store = VectorStore(
        collection_name=config.get("vector_store.collection_name", "email_embeddings"),
        persistence_dir=config.get("vector_store.persistence_dir", "./data/embeddings"),
    )
    
    # Index dataset if needed
    if vector_store.count() < len(pairs):
        print(f"  Indexing {len(pairs)} email pairs into vector store...")
        incoming_emails = [p.incoming_email for p in pairs]
        embeddings = emb_mgr.embed(incoming_emails)
        vector_store.clear()
        vector_store.add_email_pairs(pairs, embeddings)
        print(f"  Indexed {vector_store.count()} pairs")
    else:
        print(f"  Vector store already has {vector_store.count()} indexed pairs")
    
    # LLM client
    llm_config = {
        "lm_studio_url": config.get("llm.lm_studio_url"),
        "timeout": config.get("llm.timeout", 30),
        "fallback_provider": config.get("llm.fallback_provider"),
        "fallback_api_key": config.get("llm.openai_api_key") or config.get("llm.anthropic_api_key"),
    }
    llm_client = LLMClient(llm_config)
    
    # Check LM Studio connectivity
    print(f"  Checking LM Studio connectivity at '{llm_config['lm_studio_url']}'...")
    if not llm_client.check_connectivity():
        if llm_config.get("fallback_provider"):
            print(f"  WARNING: LM Studio not reachable, using fallback: {llm_config['fallback_provider']}")
        else:
            print("  ERROR: LM Studio not reachable and no fallback configured.")
            print("  Please start LM Studio and load a model, then re-run.")
            sys.exit(1)
    else:
        print("  LM Studio is reachable ✓")
    
    generator = ResponseGenerator(emb_mgr, vector_store, llm_client)
    eval_engine = EvaluationEngine(
        embedding_mgr=emb_mgr,
        llm_client=llm_client,
        thresholds=config.get("evaluation.thresholds"),
    )
    agg_engine = AggregationEngine()
    
    # -----------------------------------------------------------------------
    # 4. Generate responses for all test emails
    # -----------------------------------------------------------------------
    print(f"[4/6] Generating responses for {len(pairs)} emails...")
    
    generated_responses = []
    failed_generation = 0
    
    for i, pair in enumerate(pairs):
        try:
            result = generator.generate_response(
                incoming_email=pair.incoming_email,
                top_k=config.get("generation.top_k_examples", 3),
                max_tokens=config.get("generation.max_tokens", 500),
                temperature=config.get("generation.temperature", 0.7),
            )
            generated_responses.append((pair, result))
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
                print(f"  Generated {i+1}/{len(pairs)} responses...", end="\r")
        
        except Exception as e:
            failed_generation += 1
            logger.error(f"Generation failed for pair {pair.id}: {e}")
            generated_responses.append((pair, None))  # Mark as failed
    
    print(f"\n  Done. {len(pairs) - failed_generation} successful, {failed_generation} failed")
    
    # -----------------------------------------------------------------------
    # 5. Evaluate all responses
    # -----------------------------------------------------------------------
    print(f"[5/6] Evaluating {len(generated_responses)} responses...")
    
    evaluation_results = []
    failed_eval = 0
    
    for i, (pair, gen_result) in enumerate(generated_responses):
        if gen_result is None:
            failed_eval += 1
            continue
        
        try:
            eval_result = eval_engine.evaluate_response(
                email_pair_id=pair.id,
                incoming=pair.incoming_email,
                generated=gen_result.generated_text,
                ground_truth=pair.response,
                formality_level=pair.metadata.formality_level,
                email_type=pair.metadata.email_type,
                retrieval_quality=gen_result.retrieval_quality,
                generation_latency_ms=gen_result.generation_latency_ms,
            )
            evaluation_results.append(eval_result)
            
            if (i + 1) % 5 == 0 or (i + 1) == len(generated_responses):
                print(f"  Evaluated {i+1}/{len(generated_responses)} responses...", end="\r")
        
        except Exception as e:
            failed_eval += 1
            logger.error(f"Evaluation failed for pair {pair.id}: {e}")
    
    print(f"\n  Done. {len(evaluation_results)} evaluated, {failed_eval} failed")
    
    # -----------------------------------------------------------------------
    # 6. Aggregate results and generate report
    # -----------------------------------------------------------------------
    print("[6/6] Aggregating results and generating report...")
    
    performance = agg_engine.aggregate_results(evaluation_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(Path(output_dir) / f"evaluation_report_{timestamp}.json")
    
    agg_engine.generate_report(
        results=evaluation_results,
        performance=performance,
        output_path=report_path,
    )
    
    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Emails processed:  {len(pairs)}")
    print(f"  Responses generated: {len(pairs) - failed_generation}")
    print(f"  Responses evaluated: {len(evaluation_results)}")
    print("")
    print("QUALITY SCORES (Mean)")
    print("-" * 40)
    for dim, stats in performance.dimension_stats.items():
        print(f"  {dim:25s}: {stats['mean']:.3f}")
    print("")
    print("QUALITY DISTRIBUTION")
    print("-" * 40)
    for level, data in performance.quality_distribution.items():
        print(f"  {level:20s}: {data['count']:3d} ({data['percentage']:5.1f}%)")
    print("")
    print(f"Report saved to: {report_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
