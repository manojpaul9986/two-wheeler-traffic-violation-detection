import sys
import os
import json
from solution import TrafficViolationDetector

def main():
    # Initialize detector with bundled models
    model = TrafficViolationDetector("./models")

    # Determine image path from CLI argument or default test path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Example / Default fallback path
        image_path = "test_image.jpg"
        print(f"[INFO] No image path provided. Usage: python inference.py <path_to_image>")
        print(f"[INFO] Attempting default test path: '{image_path}'\n")

    if not os.path.exists(image_path):
        print(f"[WARNING] Image '{image_path}' not found on disk.")
        print(f"[INFO] Running predict() to demonstrate safe fallback output:")

    # Run inference
    output = model.predict(image_path)
    print("\n--- Detection Result ---")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()