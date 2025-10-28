"""Evaluation metrics for image captioning."""

import json
from typing import Dict, List
from collections import defaultdict

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.spice.spice import Spice
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: pycocoevalcap not available. Metrics will be limited.")
    METRICS_AVAILABLE = False


class CaptionEvaluator:
    """Evaluator for image captions."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.scorers = {}
        
        if METRICS_AVAILABLE:
            self.scorers['CIDEr'] = Cider()
            self.scorers['BLEU'] = Bleu(4)
            try:
                self.scorers['SPICE'] = Spice()
            except Exception as e:
                print(f"Warning: SPICE not available: {e}")
    
    def evaluate(
        self,
        predictions: Dict[int, List[str]],
        references: Dict[int, List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: Dictionary mapping image_id to list of predicted captions
            references: Dictionary mapping image_id to list of reference captions
            
        Returns:
            Dictionary of metric scores
        """
        if not METRICS_AVAILABLE:
            return self._simple_evaluation(predictions, references)
        
        # Convert to pycocoevalcap format
        # References: {image_id: [caption1, caption2, ...]}
        # Predictions: {image_id: [predicted_caption]}
        
        gts = {}  # ground truths
        res = {}  # results
        
        for img_id in predictions.keys():
            if img_id in references:
                gts[img_id] = references[img_id]
                # Predictions should be a list with single element
                if isinstance(predictions[img_id], str):
                    res[img_id] = [predictions[img_id]]
                else:
                    res[img_id] = predictions[img_id]
        
        scores = {}
        
        # Compute scores
        for metric_name, scorer in self.scorers.items():
            try:
                if metric_name == 'BLEU':
                    score, _ = scorer.compute_score(gts, res)
                    # BLEU returns scores for BLEU-1, BLEU-2, BLEU-3, BLEU-4
                    scores['BLEU-1'] = score[0]
                    scores['BLEU-2'] = score[1]
                    scores['BLEU-3'] = score[2]
                    scores['BLEU-4'] = score[3]
                else:
                    score, _ = scorer.compute_score(gts, res)
                    scores[metric_name] = score
            except Exception as e:
                print(f"Warning: Could not compute {metric_name}: {e}")
        
        return scores
    
    def _simple_evaluation(
        self,
        predictions: Dict[int, List[str]],
        references: Dict[int, List[str]]
    ) -> Dict[str, float]:
        """
        Simple evaluation metrics when pycocoevalcap is not available.
        
        Args:
            predictions: Dictionary mapping image_id to predicted caption
            references: Dictionary mapping image_id to reference captions
            
        Returns:
            Dictionary with basic metrics
        """
        # Calculate simple word overlap metric
        total_overlap = 0
        total_refs = 0
        
        for img_id in predictions.keys():
            if img_id not in references:
                continue
            
            pred = predictions[img_id]
            if isinstance(pred, list):
                pred = pred[0]
            
            pred_words = set(pred.lower().split())
            
            # Check overlap with each reference
            for ref in references[img_id]:
                ref_words = set(ref.lower().split())
                overlap = len(pred_words & ref_words)
                total_overlap += overlap
                total_refs += len(ref_words)
        
        avg_overlap = total_overlap / total_refs if total_refs > 0 else 0
        
        return {
            'word_overlap': avg_overlap,
            'note': 'Install pycocoevalcap for full metrics (CIDEr, BLEU, SPICE)'
        }
    
    def print_scores(self, scores: Dict[str, float]):
        """
        Print evaluation scores in formatted way.
        
        Args:
            scores: Dictionary of metric scores
        """
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Primary metric
        if 'CIDEr' in scores:
            print(f"CIDEr (Primary):        {scores['CIDEr']:.4f}")
        
        # BLEU scores
        if 'BLEU-4' in scores:
            print(f"BLEU-4:                 {scores['BLEU-4']:.4f}")
        if 'BLEU-1' in scores:
            print(f"BLEU-1:                 {scores['BLEU-1']:.4f}")
            print(f"BLEU-2:                 {scores['BLEU-2']:.4f}")
            print(f"BLEU-3:                 {scores['BLEU-3']:.4f}")
        
        # SPICE
        if 'SPICE' in scores:
            print(f"SPICE:                  {scores['SPICE']:.4f}")
        
        # Other metrics
        for key, value in scores.items():
            if key not in ['CIDEr', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'SPICE']:
                if isinstance(value, (int, float)):
                    print(f"{key}:                  {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        print("="*60 + "\n")


def save_predictions(
    predictions: Dict[int, str],
    output_path: str
):
    """
    Save predictions to JSON file.
    
    Args:
        predictions: Dictionary mapping image_id to predicted caption
        output_path: Path to save predictions
    """
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_path}")


def save_metrics(
    metrics: Dict[str, float],
    efficiency_metrics: Dict[str, float],
    output_path: str
):
    """
    Save all metrics to JSON file.
    
    Args:
        metrics: Evaluation metrics
        efficiency_metrics: Efficiency metrics
        output_path: Path to save metrics
    """
    all_metrics = {
        'evaluation_metrics': metrics,
        'efficiency_metrics': efficiency_metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")
