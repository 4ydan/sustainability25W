import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import random
import textwrap
import matplotlib.pyplot as plt
import config
from PIL import Image
from collections import defaultdict


def plot_eval_summary(results):
    """
    This function produces a bar plot, which shows the average values of each performance metric for each
    quantization method. It shows either 2 or 3 diagrams depending on wether SPICE is available.

    Args:
        results: results dictionairy in this format:  
        results[mode] = {
            "CIDEr": cider_mean,
            "BLEU-4": bleu_mean,
            "SPICE": spice_mean,
        }
    """
    modes = list(results.keys())  

    possible_metrics = ["CIDEr", "BLEU-4", "SPICE"]
    metrics = []
    for m in possible_metrics:
        if any(results[mode].get(m) is not None for mode in modes):
            metrics.append(m)


    number_cols = len(metrics)
    number_rows = math.ceil(len(metrics) / number_cols)

    _, axes = plt.subplots(
        nrows=number_rows,
        ncols=number_cols,
        figsize=(7*number_cols, 5 * number_rows)
    )
    axes = axes.ravel()

    for ax in axes:
        ax.set_axis_off()

    colors = plt.cm.tab10.colors

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_axis_on()
        ax.set_title(metric)

        vals = [results[mode].get(metric) for mode in modes]
        plot_vals = [v if v is not None else 0.0 for v in vals]

        color = colors[idx % len(colors)]

        ax.bar(modes, plot_vals, color=color)
        ax.set_ylabel(metric)
    

    plt.tight_layout()
    plt.show()



def plot_efficiency_pies(efficiency_metrics):
    """
    This function produces multiple pie plots, which show the vaious saved performace metrics of the models grouped by their quantization mode,
    it show peak Vram usage, latency per image, throughput and model size

    Args:
        efficiency_metrics dictionairy which contains the afore mentioned stats, for each quantization key
     
    """
    modes = list(efficiency_metrics.keys())

    metrics = {
        "Peak VRAM (MiB)": [efficiency_metrics[m]["peak_VRAM"] for m in modes],
        "Latency per Image (s)": [efficiency_metrics[m]["latency_per_image"] for m in modes],
        "Throughput (img/s)": [efficiency_metrics[m]["throughput"] for m in modes],
        "Model Size (MiB)": [efficiency_metrics[m]["model_size"] for m in modes],
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    cmap = mpl.colormaps["tab10"]

    def format_value(v):
        if v >= 1000:
            return f"{v/1000:.1f}K"
        elif v < 1:
            return f"{v:.2f}"
        else:
            return f"{v:.1f}"

    def make_pie(ax, values, title):
        wedges, texts = ax.pie(
            values,
            labels=modes,
            startangle=90,
            colors=[cmap(i) for i in range(len(modes))],
            wedgeprops=dict(width=0.35, edgecolor="w"),
            textprops=dict(color="black", fontsize=11),
        )

        
        total = sum(values)
        for w, v in zip(wedges, values):
            ang = np.deg2rad((w.theta2 + w.theta1) / 2)
            radius = (w.r + (w.r - w.r * 0.35)) / 2
            x = radius * np.cos(ang)
            y = radius * np.sin(ang)
 
            ax.text(
                 x, y, format_value(v),
                 ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white",
            )

        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("equal")

    for ax, (title, values) in zip(axes, metrics.items()):
        make_pie(ax, values, title)

    plt.suptitle("Efficiency Metrics by Quantization Mode", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.show()



def show_example_captions(
    num_examples=1,
    seed=0,
):
    """
    This function produces various images, with the associated captions. The captions consist of the Ground Truth and the captions for each quantization mode.

    Args:
        num_examples: used do determine the amount of images that should be shown
        seed: the images are picked randomly, the seed is used to determine set randomness
     
    """
 
    images_dir = config.IMAGES_DIR
    output_dir_base = config.OUTPUT_DIR
    quantizations = ["none", "skip_vision_tower", "full"]

    with open(f"{config.ANNOTATIONS_DIR}/captions_val2017.json", "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2fname = {img["id"]: img["file_name"] for img in coco["images"]}
    gts_by_stem = defaultdict(list)
    for ann in coco["annotations"]:
        fname = id2fname[ann["image_id"]]
        stem = os.path.splitext(fname)[0]
        gts_by_stem[stem].append(ann["caption"])

    preds = {}
    for mode in quantizations:
        pred_path = os.path.join(f"{output_dir_base}_{mode}", "predicted_captions.json")
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_list = json.load(f)

        mode_preds = {}
        for p in pred_list:
            img_id = p["image_id"]
            if isinstance(img_id, int):
                fname = id2fname.get(img_id)
                if fname is None:
                    continue
                stem = os.path.splitext(fname)[0]
            else:
                stem = str(img_id)
            mode_preds[stem] = p["caption"]
        preds[mode] = mode_preds

    modes = [m for m in quantizations if m in preds]

    common = set(gts_by_stem.keys())
    for mode in modes:
        common &= set(preds[mode].keys())

    rng = random.Random(seed)
    chosen = rng.sample(sorted(common), min(num_examples, len(common)))

    for stem in chosen:
        img_path = os.path.join(images_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, stem + ".png")
            if not os.path.exists(img_path):
                print(f"Image not found for {stem}, skipping.")
                continue

        img = Image.open(img_path).convert("RGB")

        fig, (ax_img, ax_txt) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [3, 2]}
        )

        ax_img.imshow(img)
        ax_img.axis("off")

      
        ax_txt.axis("off")

        gt_caption = gts_by_stem[stem][0]
        lines = [f"GT:  {gt_caption}"]
        for mode in modes:
            lines.append(f"{mode.upper()}:  {preds[mode][stem]}")

        wrapped_lines = []
        for line in lines:
            wrapped = textwrap.fill(line, width=60)
            wrapped_lines.append(wrapped)
        caption_text = "\n\n".join(wrapped_lines)  

        
        ax_txt.text(
            0.0, 0.5, caption_text,
            transform=ax_txt.transAxes,
            ha="left", va="center",
            fontsize=11,
            wrap=True,
            linespacing=1.7,
        )

        plt.tight_layout()
        plt.show()