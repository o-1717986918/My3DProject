import numpy as np
import onnxruntime as ort

def export_model(model_class, weights_path, output_file):
    """
    Export a PyTorch model to ONNX format automatically detecting input shape.
    """
    import torch  # imported only here

    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Infer input size from first Linear layer
    first_linear = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    input_size = first_linear.in_features
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Model exported to {output_file} (input size: {input_size})")


def load_network(model_path):
    """
    Load an ONNX model into memory for fast reuse.
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return {"session": session, "input_name": input_name, "output_name": output_name}


def run_network(obs, model):
    """
    Run a preloaded ONNX model and return a flat float array suitable for motor targets.

    Args:
        obs (np.ndarray): Input observation array.
        model (dict): The loaded model from load_network().

    Returns:
        np.ndarray: 1D array of floats.
    """
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    else:
        obs = obs.astype(np.float32)  # ensure float32

    if obs.ndim == 1:
        obs = obs[np.newaxis, :]  # make batch dimension

    session = model["session"]
    input_name = model["input_name"]
    output_name = model["output_name"]

    result = session.run([output_name], {input_name: obs})[0]

    # flatten to 1D and convert to float
    result = result.flatten().astype(np.float32)

    return result