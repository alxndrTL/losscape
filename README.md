# loss + landscape = losscape 🌄

This is the official repository for `losscape`, a lightweight, modular, and straightforward Python library that empowers you to visualize the loss landscape of your neural networks, as in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913).

`losscape` is designed as a plug-and-play extension for your existing PyTorch models. It boasts its simplicity, modularity, and lightning-fast rendering speed over other comparable libraries.

## Key Features ✨

- 🏃 **Fast Rendering**: Leveraging PyTorch's `torch.no_grad()` feature and controlling the number of examples to compute the loss, `losscape` speeds up the visualization process dramatically, thus saving valuable wall time.
- 📊 **Flexible Plotting**: Supports both 1D and 2D plotting options for your loss landscapes, giving you the power to choose based on your requirements and computational constraints.
- 🔌 **Plug and Play**: Easy to integrate with your existing PyTorch models. No substantial modifications to your codebase are needed!

## Installation 📦

You can install `losscape` via pip:

```
pip install losscape
```

## Usage ⚙️
Here's a quick example on how to use Losscape:


```
from losscape.train import train
from losscape.create_landscape import create_2D_losscape

model = ... # create your torch model (subclass of torch.nn.Module)
train_loader = ... # the DataLoader containing your favorite dataset

train(model, train_loader) # losscape can perform the training for you
create_2D_losscape(model, train_loader, output_vtp=True)
```

Typical results :


To visualize the loss landscape in 3D, use the `.vtp` file created by the library and simply drag-and-drop it in [a VTK viewer](https://kitware.github.io/itk-vtk-viewer/app/) :

## Documentation 📖
For more details, please refer to the documentation. It provides a global overview on how `losscape` works, and on how you can leverage it with your own model.

## Contact 📞
Feel free to open an issue if you find a bug or have any suggestions to improve the library. Feedback is much appreciated!

# Enjoy the visualization journey with Losscape! 🎉
