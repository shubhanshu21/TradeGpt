"""
SOVEREIGN KRAKEN (V10.4) - RECURSIVE RISK SUPERVISOR 🛡️🔄
==========================================================
Self-Refining Risk Module: Adjusts thresholds based on expert consensus.
"""

import numpy as np

class PredatorRiskManager:
    def __init__(self, base_thresh=0.15, min_thresh=0.08, max_thresh=0.35):
        self.base_thresh = base_thresh
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.current_thresh = base_thresh
        self.consensus_history = []

    def evaluate_threshold(self, consensus_score):
        """
        Adjusts the Z-Score threshold based on Expert Consensus.
        - High Consensus (0.9+) -> Lower threshold (Strike more often)
        - Low Consensus (<0.6)  -> Higher threshold (Safety mode)
        """
        self.consensus_history.append(consensus_score)
        if len(self.consensus_history) > 60:
            self.consensus_history.pop(0)

        avg_consensus = np.mean(self.consensus_history)

        # Logic: Threshold is inversely proportional to consensus
        # If consensus is high (1.0), multiplier is low.
        multiplier = 1.0 + (0.8 - avg_consensus) * 2.0
        new_thresh = self.base_thresh * multiplier
        
        # Clamp to safety bounds
        self.current_thresh = np.clip(new_thresh, self.min_thresh, self.max_thresh)
        
        return self.current_thresh

    def get_tactical_status(self):
        if self.current_thresh > self.base_thresh * 1.5:
            return "🛡️ SAFETY MODE ( Experts Confused )"
        elif self.current_thresh < self.base_thresh * 0.8:
            return "🔥 AGGRESSION MODE ( High Alpha )"
        return "⚖️ NOMINAL MODE"
