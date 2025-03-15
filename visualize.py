import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, scores, prompt):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, idx in zip(axes, images, range(len(images))):
        ax.imshow(img)
    plt.suptitle(f"Prompt: {prompt}\n"
                 f"CLIP Diversity: {scores['clip_mean']:.3f} ± {scores['clip_std']:.3f}\n"
                 f"Entropy: {scores['entropy']:.3f}")
    plt.show()