import time

import timm
import torch
import torchvision.transforms as transforms
from memory_profiler import memory_usage
from PIL import Image
from threadpoolctl import threadpool_limits


def process_and_predict(image_path, model_path):
    model = timm.create_model(model_path, pretrained=True)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities


def benchmark_model(image_path, model_name, N=10):
    times = []
    peak_memory_usages = []

    for _ in range(N):
        start_time = time.time()

        # Measure peak memory usage
        peak_memory_usage = memory_usage(
            (process_and_predict, (image_path, model_name)),
            interval=0.01,
            max_usage=True,
            include_children=True,
        )

        end_time = time.time()

        time_taken = end_time - start_time
        times.append(time_taken)
        peak_memory_usages.append(peak_memory_usage)

    avg_time = sum(times) / N * 1000  # in ms
    max_peak_memory = sum(peak_memory_usages) / N
    return avg_time, max_peak_memory


# model variants
model_variants = {
    "tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "large": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}

# an image
image_path = "./assets/tench.jpg"

if __name__ == "__main__":
    with threadpool_limits(limits=4):
        print("| Model | Speed (ms)   |   Mem (MB)       |")
        print("|-------|--------------|------------------|")

        for name, model_name in model_variants.items():
            avg_time, peak_memory = benchmark_model(image_path, model_name)
            print(f"| {name} | {avg_time:.0f} | {peak_memory:.0f} |")
