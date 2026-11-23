# Explainable AI for Pneumonia Detection

A CNN that classifies chest X-rays as normal or pneumonia, paired with a LIME-inspired perturbation method that shows which pixels drove each prediction. The goal is a model whose outputs a clinician can check rather than take on faith.

Paper: [Research Archive preprint](https://research-archive.org/index.php/rars/preprint/view/1724/2453)

## How it works

1. **Classifier**: a CNN built in PyTorch and trained on [PneumoniaMNIST](https://medmnist.com/) (5,856 grayscale chest X-rays, 28×28).
2. **Explanation**: pixels are perturbed and the change in the model's output is measured. Pixels that shift the prediction most are ranked as most important.
3. **Visualization**: importance scores are rendered as a heatmap and overlaid on the original X-ray.

The explanation method only needs the model's predictions, so it can be used with other classifiers too.

## Results

- ~90% test accuracy after 15 epochs.
- Heatmaps highlight the lung regions associated with pneumonia.
- Misclassified scans tend to show unusual importance patterns, which makes errors easier to spot.

## Example

| Original X-ray | Pixel importance | Overlay |
|---|---|---|
| ![Original X-ray](images/originalchest.png) | ![Heatmap](images/heatmap.png) | ![Overlay](images/overlayed.png) |

## Running it

1. Open the notebook in Google Colab.
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type).
3. Run all cells.

Built with Python, PyTorch 2.1 and CUDA 11.8.
