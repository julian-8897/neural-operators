"""Data loading utilities for 2D Darcy Flow."""

from neuralop.datasets import load_darcy_flow_small


def get_darcy_flow_dataloaders(config):
    """
    Load Darcy Flow dataset using neuraloperator's built-in dataset.

    Args:
        config: Configuration object with data parameters

    Returns:
        train_loader, test_loader, data_processor
    """
    # Load the built-in Darcy Flow dataset
    # This automatically downloads and caches the dataset
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=config.train_samples,
        n_tests=[config.test_samples],
        batch_size=config.batch_size,
        test_batch_sizes=[config.test_batch_size],
        test_resolutions=[config.train_resolution],
        grid_boundaries=config.grid_boundaries,
        positional_encoding=True,
        encode_input=True,
        encode_output=True,
        num_workers=0,  # Set to 0 for Mac compatibility
        persistent_workers=False,
    )

    # test_loaders is a dict with resolution as key
    test_loader = test_loaders[config.train_resolution]

    return train_loader, test_loader, data_processor


def visualize_sample(
    input_tensor, output_tensor, prediction=None, data_processor=None, save_path=None
):
    """
    Visualize a sample from the Darcy Flow dataset.

    Args:
        input_tensor: Input permeability field
        output_tensor: Ground truth solution
        prediction: Model prediction (optional)
        data_processor: Data processor for denormalization
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt

    # Decode if processor is provided
    if data_processor is not None:
        input_tensor = data_processor.decode(input_tensor)
        output_tensor = data_processor.decode(output_tensor)
        if prediction is not None:
            prediction = data_processor.decode(prediction)

    # Move to CPU and convert to numpy
    input_np = input_tensor.squeeze().cpu().numpy()
    output_np = output_tensor.squeeze().cpu().numpy()

    n_plots = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Input (permeability field)
    im0 = axes[0].imshow(input_np, cmap="viridis")
    axes[0].set_title("Input: Permeability Field")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0])

    # Ground truth (pressure field)
    im1 = axes[1].imshow(output_np, cmap="coolwarm")
    axes[1].set_title("Ground Truth: Pressure Field")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    # Prediction
    if prediction is not None:
        pred_np = prediction.squeeze().cpu().numpy()
        im2 = axes[2].imshow(pred_np, cmap="coolwarm")
        axes[2].set_title("Prediction: Pressure Field")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()
