# run this as a quick one-off script: check_weights.py
import h5py

with h5py.File("skin_phase21_final.h5", "r") as f:
    print("Top-level keys:", list(f.keys()))
    if "layer_names" in f.attrs:
        print("Layer names:", list(f.attrs["layer_names"]))
    # Print every layer and its weight shapes
    def print_weights(name, obj):
        if hasattr(obj, 'shape'):
            print(f"  {name}: {obj.shape}")
    f.visititems(print_weights)
    