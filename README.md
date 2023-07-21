# loss + landscape = `losscape` 🌄

This is `losscape`, a lightweight, modular, and straightforward Python library that empowers you to visualize the loss landscape of your neural networks, as in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913).

`losscape` is designed as a plug-and-play extension for your existing PyTorch models. It boasts its simplicity, modularity, and fast rendering speed.

## Key Features ✨

- 🏃 **Fast Rendering**: Leveraging PyTorch's `torch.no_grad()` feature and controlling the number of examples to compute the loss, `losscape` speeds up the visualization process dramatically, thus saving valuable wall time.
- 📊 **Flexible Plotting**: Supports both 2D and 3D plotting options for your loss landscapes, giving you the power to choose based on your requirements and computational constraints.
- 🔌 **Plug and Play**: Easy to integrate with your existing PyTorch models. No substantial modifications to your codebase are needed!

## Installation 📦

You can install `losscape` via pip:

```
pip install losscape
```

## Usage ⚙️
Here's a quick example on how to use `losscape`:


```
from losscape.train import train
from losscape.create_landscape import create_2D_losscape

model = ... # create your torch model (subclass of torch.nn.Module)
train_loader = ... # the DataLoader containing your favorite dataset

train(model, train_loader) # losscape can perform the training for you
create_2D_losscape(model, train_loader, output_vtp=True)
```

Typical results :

<p float="left" align="center">
  <img src="https://github.com/Procuste34/losscape/blob/main/docs/1d_landscape.png?raw=true" width="400" />
  <img src="https://github.com/Procuste34/losscape/blob/main/docs/2d_landscape.png?raw=true" width="400" /> 
</p>

To visualize the loss landscape in 3D, use the `.vtp` file created by the library and simply drag-and-drop it in [a VTK viewer](https://kitware.github.io/itk-vtk-viewer/app/) :

<p align="center">
  <img src="https://github.com/Procuste34/losscape/blob/main/docs/3d_landscape.png?raw=true" width="300" />
</p>

## Documentation 📖
For more details, please refer to the documentation. It provides a global overview on how `losscape` works, and on how you can leverage it with your own model.
(to be written : for now you can refer to the code, its simple enough to understand the structure of the library, and you can spin up using the juypter notebook example, all available on the Github repo).

## References 📚
- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- [Visualizing the Loss Landscape of Winning Lottery Tickets](https://arxiv.org/abs/2112.08538)


## Roadmap 🚀
Here are a couple of additions I'm planning to add in the near future:

- 📈 Validation Loss Landscape: I'm working on the ability to visualize the validation loss landscape, in addition to the training loss landscape.
- 🛤 Optimizer Path Visualization: Future updates will include the ability to visualize the path of gradient descent (or any other optimizer) on these landscapes using PCA.

## Contact 📞
Feel free to open an issue if you find a bug or have any suggestions to improve the library. Feedback is much appreciated!

# Enjoy your visualization journey with `losscape` ! 🎉
