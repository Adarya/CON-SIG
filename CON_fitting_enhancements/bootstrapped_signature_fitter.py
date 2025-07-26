import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

# Import the base fitter from the existing CON_fitting framework
from CON_fitting.src.signature_fitter import ConsensusSignatureFitter


class BootstrappedSignatureFitter:
    """Estimate signature activities with bootstrap-based confidence intervals.

    This wrapper repeatedly resamples the input CNA matrix to generate an
    empirical distribution of signature activities, thus providing
    uncertainty estimates (mean / 95 % CI) in addition to the point
    estimates returned by ``ConsensusSignatureFitter``.
    """

    def __init__(
        self,
        consensus_signatures: pd.DataFrame,
        n_iterations: int = 200,
        method: str = "nnls",
        sample_fraction: float = 1.0,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ) -> None:
        """Parameters
        ----------
        consensus_signatures
            DataFrame (rows = CNA categories, cols = signatures).
        n_iterations
            Number of bootstrap replicates.
        method
            Deconvolution algorithm to pass to ``ConsensusSignatureFitter``.
        sample_fraction
            Fraction of CNA events (columns) to sample with replacement for
            each replicate (``1.0`` = classic bootstrap over all columns).
        random_state
            Seed for NumPy RNG (ensures reproducibility).
        verbose
            Whether to print progress information.
        """
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.verbose = verbose

        # Re-use the existing, well-tested fitter for the actual NNLS/EN etc.
        self._base_fitter = ConsensusSignatureFitter(
            consensus_signatures=consensus_signatures,
            method=method,
            verbose=verbose,
        )

        # RNG
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _bootstrap_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a bootstrap replicate of *data* by resampling CNA categories.

        The classic bootstrap samples columns (CNA categories) with
        replacement.  The resulting matrix preserves the original shape but
        re-weights categories, which captures variance in the exposure
        estimation due to finite event counts per category.
        """
        n_categories = data.shape[1]
        n_sample = int(round(n_categories * self.sample_fraction))
        idx = self._rng.choice(n_categories, size=n_sample, replace=True)
        # Use iloc for speed; aggregate duplicates by summing
        boot = data.iloc[:, idx].copy()
        # If some categories are sampled multiple times, their counts add up
        boot = boot.groupby(level=0, axis=1).sum()
        # Re-index to full set, filling missing categories with zeros
        boot = boot.reindex(columns=data.columns, fill_value=0)
        return boot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit signatures and compute bootstrap confidence intervals.

        Parameters
        ----------
        data
            CNA matrix with samples as rows and CNA categories as columns.

        Returns
        -------
        point_estimates
            Signature activities from the original data (no resampling).
        base_metrics
            Quality metrics returned by the base fitter.
        mean_activities
            Mean of bootstrap activities across replicates.
        ci_lower
            Lower bound (2.5 percentile) of bootstrap distribution.
        ci_upper
            Upper bound (97.5 percentile) of bootstrap distribution.
        """
        if self.verbose:
            print(
                f"Bootstrapping signature activities: {self.n_iterations} iterations, "
                f"sample_fraction={self.sample_fraction}"
            )

        # ------------------------------------------------------------------
        # 1. Point estimate on the full data
        # ------------------------------------------------------------------
        point_estimates, base_metrics = self._base_fitter.fit(data)

        # Prepare containers for bootstrap distributions
        n_samples, n_signatures = point_estimates.shape
        boot_array = np.zeros((self.n_iterations, n_samples, n_signatures))

        for i in range(self.n_iterations):
            # Resample data
            boot_data = self._bootstrap_resample(data)
            # Fit
            activities_i, _ = self._base_fitter.fit(boot_data)
            boot_array[i] = activities_i.values

            if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                print(f"  ▸ Completed {i + 1}/{self.n_iterations} iterations")

        # ------------------------------------------------------------------
        # 2. Aggregate bootstrap results
        # ------------------------------------------------------------------
        mean_activities = pd.DataFrame(
            boot_array.mean(axis=0),
            index=point_estimates.index,
            columns=point_estimates.columns,
        )
        ci_lower = pd.DataFrame(
            np.percentile(boot_array, 2.5, axis=0),
            index=point_estimates.index,
            columns=point_estimates.columns,
        )
        ci_upper = pd.DataFrame(
            np.percentile(boot_array, 97.5, axis=0),
            index=point_estimates.index,
            columns=point_estimates.columns,
        )

        return point_estimates, base_metrics, mean_activities, ci_lower, ci_upper 