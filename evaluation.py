import json
import config
from collections import defaultdict

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.spice.spice import Spice

def import_data(quantization_mode):
    with open(f"{config.ANNOTATIONS_DIR}/captions_val2017.json", "r") as f:
        actual_captions_raw = json.load(f)["annotations"]

    with open(f"{config.OUTPUT_DIR}_{quantization_mode}/predicted_captions.json" , "r") as f:
        predicted_captions_raw = json.load(f)

    actual_captions = defaultdict(list)
    for ann in actual_captions_raw:
        actual_captions[ann["image_id"]].append(ann["caption"])

    predicted_captions = defaultdict(list)
    for p in predicted_captions_raw:
        img_id = int(str(p["image_id"]).lstrip("0"))  
        predicted_captions[img_id].append(p["caption"])

    common_ids = actual_captions.keys() & predicted_captions.keys()

    actual_captions = {i: actual_captions[i] for i in common_ids}
    predicted_captions = {i: predicted_captions[i] for i in common_ids}

    return (actual_captions, predicted_captions)


def calculate_cider_score(actual_captions, predicted_captions):
    scorer = Cider()
    score, scores = scorer.compute_score(actual_captions, predicted_captions)
    return score, scores


def calculate_bleu_score(actual_captions, predicted_captions):
    scorer = Bleu(4)  
    score, scores = scorer.compute_score(actual_captions, predicted_captions)
    bleu4 = score[3]  
    return bleu4, scores[3]


def calculate_spice_score(actual_captions, predicted_captions):
    try:
        scorer = Spice()
        score, scores = scorer.compute_score(actual_captions, predicted_captions)
        return score, scores
    except FileNotFoundError as e:
        print(f"SPICE could not be executed: {e}")
        return None, None


def load_metrics(quantization_mode):
    with open(f"{config.OUTPUT_DIR}_{quantization_mode}/metrics.json" , "r") as f:
        return json.load(f)
        
