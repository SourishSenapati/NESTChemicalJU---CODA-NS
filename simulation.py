
"""
Simulation module for generating synthetic clinical trial data.
"""
import logging
import traceback
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogicValidator:
    """
    Enforces deterministic logical locking for the Zero-Error architecture.
    """

    @staticmethod
    def validate_row(row):
        """
        Checks for impossible states in a clinical data row.
        Returns True if row is valid, False otherwise.
        """
        # Rule 1: Cannot have SAEs if ReviewStatus is N/A or None, unless count is 0
        if row['SAE_Count'] > 0 and row['ReviewStatus'] in [None, 'N/A', 'nan']:
            return False

        return True

    @staticmethod
    def enforce_determinism(data_df):
        """
        Applies post-hoc logical locking to fix inconsistencies and derive calculated fields.
        """
        data_df = data_df.copy()

        # 1. SAE Consistency - STRICT CONSTRAINT (Sub-Task 1.2)
        # If SAE_Count > 0 but Review_Status is N/A/empty, DELETE the row.
        mask_invalid_sae = (data_df['SAE_Count'] > 0) & (
            data_df['ReviewStatus'].isin(['N/A', None, 'nan']))

        if mask_invalid_sae.any():
            deleted_count = mask_invalid_sae.sum()
            logger.info(
                "Logic Filter: Dropping %d rows with SAE > 0 "
                "but no Review Status.",
                deleted_count
            )
            data_df = data_df[~mask_invalid_sae]

        # If SAE == 0, force status to 'N/A'
        # (This is just data cleaning, not the critical delete constraint)
        mask_no_sae = data_df['SAE_Count'] == 0
        data_df.loc[mask_no_sae, 'ReviewStatus'] = 'N/A'

        # 2. Logic Locking: Calculated Clean Status
        # The Golden Truth: A patient is "Clean" (Fit for analysis) IF:
        # - No Missing Visits
        # - No Open Queries
        data_df['Calculated_Clean_Status'] = (
            (data_df['MissingVisits'] == 0) &
            (data_df['OpenQueries'] == 0)
        ).map({True: 'TRUE', False: 'FALSE'})

        return data_df


class ClinicalDataSimulator:
    """
    Hyper-Rigorous Simulation Engine for CODN-350M Training Data.
    Uses Gaussian Copula for statistical correlations and enforces Deterministic Logic Locking.
    """

    def __init__(self, n_samples: int = 50000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        self.audit_log = []

    def _log_action(self, action: str):
        """Forensic logging of simulation steps."""
        timestamp = datetime.now().isoformat()
        action_hash = hashlib.sha512(
            f"{action}{timestamp}".encode()).hexdigest()
        self.audit_log.append({
            "timestamp": timestamp,
            "action": action,
            "hash": action_hash
        })
        logger.info("ACTION: %s | HASH: %s...", action, action_hash[:16])

    def generate_seed_data(self) -> pd.DataFrame:
        """
        Generates a seed DataFrame for SDV to learn distributions from.
        This represents the 'Raw' observations before synthesis.
        """
        self._log_action("Generating Seed Data for SDV")

        # Create a smaller seed (e.g., 20% of target or fixed size) to fit the Copula
        seed_size = min(5000, self.n_samples // 10)

        data = {}

        # 1. Site Context (Categorical)
        n_sites = 50
        data['SiteID'] = [
            f"SITE-{np.random.randint(1, n_sites+1):03d}" for _ in range(seed_size)]
        data['Region'] = np.random.choice(
            ['North America', 'EMEA', 'APAC', 'LATAM'], seed_size)

        # 2. Queries & Visits (Numerical)
        data['MissingVisits'] = np.random.poisson(0.5, seed_size)
        data['OpenQueries'] = np.random.poisson(1.5, seed_size)
        data['QueryType'] = np.random.choice(
            ['Safety', 'Efficacy', 'Admin', 'None'],
            seed_size,
            p=[0.1, 0.1, 0.1, 0.7]
        )

        # Clean up QueryType
        data['QueryType'] = np.where(
            data['OpenQueries'] == 0, 'None', data['QueryType'])
        data['QueryType'] = np.where(
            (data['OpenQueries'] > 0) & (data['QueryType'] == 'None'),
            'Admin',
            data['QueryType']
        )

        # 3. Safety Signals
        data['SAE_Count'] = np.random.poisson(0.1, seed_size)
        data['ReviewStatus'] = np.where(
            data['SAE_Count'] > 0,
            np.random.choice(['Pending', 'Reviewed', 'Escalated'], seed_size),
            'N/A'
        )

        seed_df = pd.DataFrame(data)

        # Enforce types
        seed_df['MissingVisits'] = seed_df['MissingVisits'].astype(int)
        seed_df['OpenQueries'] = seed_df['OpenQueries'].astype(int)
        seed_df['SAE_Count'] = seed_df['SAE_Count'].astype(int)

        return seed_df

    def load_and_validate_csv(self, file_path: str) -> pd.DataFrame:
        """
        Sub-Task 1.1: Load real data and validate against schema.
        """
        self._log_action(f"Loading Real Data from {file_path}")
        if not traceback.sys.modules.get("os").path.exists(file_path):
            raise FileNotFoundError(f"CSV not found: {file_path}")

        df = pd.read_csv(file_path)
        # Basic validation (Check specific columns exist)
        required_cols = ['SiteID', 'SubjectID',
                         'MissingVisits', 'OpenQueries', 'SAE_Count']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure Logic Locking on Real Data
        return LogicValidator.enforce_determinism(df)

    def synthesis_pipeline(self, seed_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Executes the full SDV synthesis and logic locking pipeline.
        If seed_df is provided, fits to that (Sub-Task 1.2). 
        Otherwise generates synthetic seed.
        """
        try:
            # 1. Generate/Load Seed
            if seed_df is None:
                seed_df = self.generate_seed_data()

            self._log_action(f"Seed Data ready: {seed_df.shape}")

            # Debugging: Check for NaN
            if seed_df.isnull().any().any():
                logger.warning(
                    "NaN values found in seed data! Filling with defaults.")
                seed_df = seed_df.fillna(0)  # Basic fill for safety

            # 2. Define Metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(seed_df)

            # Refine Metadata
            # Ensure categorical columns are treated as such
            for col in ['SiteID', 'Region', 'QueryType', 'ReviewStatus']:
                if col in seed_df.columns:
                    metadata.update_column(
                        column_name=col, sdtype='categorical')

            self._log_action("Metadata Defined. Fitting Synthesizer...")

            # 3. Fit Gaussian Copula
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(seed_df)
            self._log_action("Synthesizer Fit Complete.")

            # 4. Generate Synthetic Data
            self._log_action(f"Sampling {self.n_samples} rows...")
            synthetic_df = synthesizer.sample(num_rows=self.n_samples)

            # 5. Post-Processing & Logic Locking
            self._log_action("Applying Deterministic Logic Locking...")

            # Add SubjectIDs (unique for the new dataset)
            synthetic_df['SubjectID'] = [
                f"SUBJ-{i:06d}" for i in range(len(synthetic_df))]

            # Enforce Logic
            locked_df = LogicValidator.enforce_determinism(synthetic_df)

            # Reorder columns for narrative consistency (if columns exist)
            col_order = [
                'Timepoint',
                'SiteID',
                'Region',
                'SubjectID',
                'MissingVisits',
                'OpenQueries',
                'QueryType',
                'SAE_Count',
                'ReviewStatus',
                'Calculated_Clean_Status'
            ]

            # Ensure Timepoint exists
            if 'Timepoint' not in locked_df.columns:
                locked_df['Timepoint'] = "T(0)"

            # Filter only existing columns
            existing_cols = [c for c in col_order if c in locked_df.columns]
            locked_df = locked_df[existing_cols]

            self._log_action(
                f"Pipeline Complete. Generated {len(locked_df)} valid records.")
            return locked_df

        except Exception as e:
            logger.error("Critical Simulation Failure: %s", str(e))
            traceback.print_exc()
            raise

    def serialize_dataset(self, data_df: pd.DataFrame, output_path: str):
        """
        Converts the tabular dataframe into the prompt-engineered narrative text.
        """
        self._log_action("Serializing Dataset to Narrative Text")

        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in data_df.iterrows():
                # Logic for Critical Tokens
                if row['Calculated_Clean_Status'] == 'TRUE':
                    status_token = "[CLEAN]"
                else:
                    status_token = "[RISK] [ACTION_REQUIRED]"
                bottleneck_str = " [BOTTLENECK]" if row['OpenQueries'] > 0 else ""

                narrative = (
                    f"Timepoint {row['Timepoint']}. "
                    f"Site {row['SiteID']} context: Region {row['Region']}. "
                    f"Status Report: Subject {row['SubjectID']} has "
                    f"{row['MissingVisits']} missed visits "
                    f"and {row['OpenQueries']} open queries{bottleneck_str} "
                    f"of type {row['QueryType']}. "
                    f"Safety Signal: {row['SAE_Count']} events in "
                    f"Review Status {row['ReviewStatus']}. "
                    f"[LOGIC_GATE]: Therefore, Status: {status_token}."
                )
                f.write(narrative + "\n")

        self._log_action(f"Serialization complete: {output_path}")


if __name__ == "__main__":
    # Test run
    sim = ClinicalDataSimulator(n_samples=100)
    final_df = sim.synthesis_pipeline()
    print(final_df.head())
    sim.serialize_dataset(final_df, "codn_training_data.txt")
