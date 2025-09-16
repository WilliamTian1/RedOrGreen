"""ONNX export stub for future accelerated inference.

Use skl2onnx or hummingbird to export sklearn models, and ORT runtime to serve.
This module is a placeholder with guidance.
"""

def export_to_onnx_sklearn(model, feature_names, save_path: str) -> None:
    """Example stub.

    In a full implementation:
    - from skl2onnx import convert_sklearn
    - from skl2onnx.common.data_types import FloatTensorType
    - initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    - onnx_model = convert_sklearn(model, initial_types=initial_type)
    - with open(save_path, "wb") as f: f.write(onnx_model.SerializeToString())
    """
    raise NotImplementedError("ONNX export not implemented in this demo.")




