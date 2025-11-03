"""Main entry point for running different model configurations."""

import click

from preprocess import download_coco
from model_utils import load_model
from inference import run_inference


@click.command()
@click.option(
    "--quantization", "-q",
    type=click.Choice(["none", "skip_vision_tower", "full"], case_sensitive=False),
    default="none",
    help="Quantization mode: 'none' (baseline), 'skip_vision_tower', or 'full'",
)
@click.option(
    "--num-images", "-n",
    type=int,
    default=1,
    help="Number of images to process (use 0 for all images)",
)
@click.option(
    "--device", "-d",
    type=click.Choice(["auto", "cuda", "cpu"], case_sensitive=False),
    default="auto",
    help="Device to use: 'auto', 'cuda', or 'cpu'",
)
@click.option(
    "--save-captions",
    is_flag=True,
    help="Save captions to disk",
)
def main(quantization, num_images, device, save_captions):
    """Run model based on quantization mode configuration."""
    # Ensure dataset is downloaded
    download_coco()

    # Convert num_images: 0 means all images (None)
    num_images = None if num_images == 0 else num_images

    print("\n" + "=" * 80)
    print(f"RUNNING MODEL - QUANTIZATION MODE: {quantization.upper()}")
    print(f"Device: {device}, Images: {num_images or 'all'}, Save: {save_captions}")
    print("=" * 80 + "\n")

    # Load model with specified configuration
    processor, model, device_name, dtype = load_model(
        device_config=device,
        quantization_mode=quantization
    )

    # Run inference
    run_inference(
        processor=processor,
        model=model,
        device=device_name,
        dtype=dtype,
        num_images=num_images,
        save_captions=save_captions,
    )


if __name__ == "__main__":
    main()
