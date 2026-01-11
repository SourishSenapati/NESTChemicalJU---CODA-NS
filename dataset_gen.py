"""
Dataset Generation Script for CODN-350M.
Implements Phase 1 (Data Foundry) of the architecture.
"""
import os
import sys
import json
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from CODN_350M.simulation import ClinicalDataSimulator

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetGen")


def train_and_save_tokenizer(corpus_path, output_path="critical_tokens.json"):
    """
    Sub-Task 2.2: The Weighted Tokenizer & Critical Token Extraction
    """
    logger.info("Training Tokenizer on %s...", corpus_path)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "[LOGIC_GATE]", "TRUE", "FALSE",
        "[RISK]", "[BOTTLENECK]", "[CLEAN]", "[ACTION_REQUIRED]"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    tokenizer.train([corpus_path], trainer)

    # Save tokenizer for model Loading
    tokenizer.save("codn_tokenizer.json")
    logger.info("Tokenizer saved to codn_tokenizer.json")

    # Metrics Extraction
    critical_ids = {}
    for token in ["[RISK]", "[CLEAN]", "[BOTTLENECK]", "[ACTION_REQUIRED]"]:
        tid = tokenizer.token_to_id(token)
        if tid is not None:
            critical_ids[token] = tid

    # Add numbers 0-100 if present (common in clinical stats)
    for i in range(101):
        s_i = str(i)
        tid = tokenizer.token_to_id(s_i)
        if tid is not None:
            critical_ids[f"NUM_{i}"] = tid

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(critical_ids, f, indent=4)

    logger.info("Critical Token IDs saved to %s", output_path)


def main():
    """
    Main execution flow for dataset generation.
    """
    simulator = ClinicalDataSimulator(n_samples=500000)  # Sub-task 1.2 target

    # Sub-Task 1.1: Semantic Serialization of Real Data
    real_csv_path = os.path.join(os.path.dirname(
        __file__), "..", "..", "CPID_EDC_Metrics.csv")
    corpus_file = "synthetic_corpus_v1.txt"

    real_df = None
    if os.path.exists(real_csv_path):
        logger.info("Found real data at %s", real_csv_path)
        real_df = simulator.load_and_validate_csv(real_csv_path)

        # Serialize Real Data
        simulator.serialize_dataset(real_df, "temp_real_corpus.txt")
        logger.info("Real data serialized.")
    else:
        logger.warning(
            "CPID_EDC_Metrics.csv not found at %s. Proceeding with synthetic seed.",
            real_csv_path
        )
        # Fallback: Generate seed data internally
        real_df = simulator.generate_seed_data()

    # Sub-Task 1.2: Synthetic Expansion
    logger.info("Starting Synthetic Expansion...")
    synthetic_df = simulator.synthesis_pipeline(seed_df=real_df)

    # Save Synthetic Data
    simulator.serialize_dataset(synthetic_df, "temp_synthetic_corpus.txt")

    # Combine Corpus
    logger.info("Combining into %s...", corpus_file)
    with open(corpus_file, 'w', encoding='utf-8') as outfile:
        # 1. Real Data (if any specific file existed)
        if os.path.exists("temp_real_corpus.txt"):
            with open("temp_real_corpus.txt", 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

        # 2. Synthetic Data
        with open("temp_synthetic_corpus.txt", 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())

    logger.info("Corpus generated: %s", corpus_file)

    # Sub-Task 2.2: Tokenizer
    train_and_save_tokenizer(corpus_file, "critical_tokens.json")

    # Cleanup temp files
    if os.path.exists("temp_real_corpus.txt"):
        os.remove("temp_real_corpus.txt")
    if os.path.exists("temp_synthetic_corpus.txt"):
        os.remove("temp_synthetic_corpus.txt")


if __name__ == "__main__":
    main()
