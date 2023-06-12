# Feature Visualization:
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Assuming model is your trained model
conv_layer = model.conv1
weights = conv_layer.weight.detach().cpu()

# Normalize to (0,1) for visualization
weights = (weights - weights.min()) / (weights.max() - weights.min())

# Create a grid of images and convert to numpy
img_grid = make_grid(weights, nrow=8).numpy()

# Transpose to correct format and show
plt.imshow(np.transpose(img_grid, (1,2,0)))
plt.show()


# Activation Maps:
from torch.autograd import Variable

# Assuming model is your trained model
# Assuming img is your input image
img = Variable(img.unsqueeze(0))

feature_maps = model.conv1(img)

# Visualize the first 10 feature maps
for i in range(10):
    plt.imshow(feature_maps[0,i].detach().numpy(), cmap='hot')
    plt.show()


# t-SNE Visualization:
from sklearn.manifold import TSNE

# Assuming features is a tensor containing the features of your dataset
features = features.detach().numpy()

tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

plt.scatter(features_2d[:,0], features_2d[:,1])
plt.show()


# Grad-CAM:
from captum.attr import LayerGradCam

# Assuming model is your trained model
# Assuming input is your input tensor
# Assuming target_class is the class you want to visualize

grad_cam = LayerGradCam(model, model.conv1)
cam = grad_cam.attribute(input, target=target_class)

# cam is a tensor containing the class activation map
# You can convert it to an image and display it using plt.imshow



# Deep Dream:

from torchvision.transforms import ToPILImage
from torch.autograd import Variable

# Assuming model is your trained model
# Assuming img is your input image

model.eval()

img = Variable(img.unsqueeze(0), requires_grad=True)

for _ in range(20):  # Number of steps
    model.zero_grad()
    out = model(img)  # Forward pass
    loss = out.norm()  # Define a loss. Here we use the L2 norm of the output
    loss.backward()  # Backward pass
    img.data = img.data + 0.1 * img.grad.data  # Update the image

# Convert the tensor to an image and display it
ToPILImage()(img.squeeze()).show()

