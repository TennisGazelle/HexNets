import pathlib


def load_reference_image(n, r, image_type):
    """Load a reference image from the reference directory."""
    reference_dir = pathlib.Path("reference").resolve()

    if image_type == "structure":
        filename = f"hexnet_n{n}_r{r}_structure.png"
    elif image_type == "activation":
        filename = f"hexnet_n{n}_r{r}_Activation_Structure.png"
    elif image_type == "weight":
        filename = f"hexnet_n{n}_r{r}_Weight_Matrix.png"
    else:
        return None

    image_path = reference_dir / filename
    if image_path.exists():
        return str(image_path)
    return None


def load_multi_activation_image(n):
    """Load a multi-activation overlay image from the reference directory."""
    reference_dir = pathlib.Path("reference").resolve()
    filename = f"hexnet_n{n}_multi_activation.png"
    image_path = reference_dir / filename
    if image_path.exists():
        return str(image_path)
    return None
