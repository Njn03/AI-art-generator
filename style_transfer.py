# importing all libraries
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# imsize = 512 if torch.cuda.is_available() else 256
imgSize = 128  # to make it work on CPU faster

# loader -> pipeline of transformations = resize input image, convert from PIL object to tensor
# this is the format the model understands (also scales pixel values from [0,255] to [0,1])
loader = transforms.Compose([transforms.Resize((imgSize, imgSize)), transforms.ToTensor()])

# convert image to rgb and applies loader. Also add and extra dimension batch_size (that's what pytorch model expects)
# [batch_size, channels, height, width]
# here, batch_size=1
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# defining loss -> content loss and style loss
# __init__ = stores feature map of content image
# .detach() = because it's fixed (we dont need gradients for it during optimization)
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    # computes mse bw feature maps of generated image and content image
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input
    

# style loss -> using gram matrix = computes correlations for style instead of actual pixel
# calculate gram matrix of feature map
def gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b, c*d)
    G = torch.mm(features, features.t()) #matrix mul of feature with its transpose
    return G.div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
# pre-trained model


# normalization of input images
# per channel mean and standard deviation of imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        return (img - self.mean) / self.std


# constructing model by inserting style/content loss at selected layers
def get_style_model_n_losses(cnn, mean, std, style_img, content_img, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(mean, std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognised layer: {layer.__class__.__name__}')
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}',content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break

    model = model[:(i+1)]
    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, mean, std, content_img_path, style_img_path, output_img_path,
                       num_steps=1000, style_weight=1000000, content_weight=1):
    # we will call this function from our API
    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)
    output_img = content_img.clone() #initialize output with content image

    print('building style tranfer model...')
    model, style_losses, content_losses = get_style_model_n_losses(cnn, mean, std, style_img, content_img)
    optimizer = get_input_optimizer(output_img)

    print('optimizing...')
    run=[0]
    while run[0] <= num_steps:
        def closure():
            output_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(output_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
  
            if run[0] % 50 == 0:
                print(f'run_{run[0]}:')
                print(f'style loss: {style_score.item():.4f} | content loss: {content_score.item():.4f}')
                print(f'total loss: {loss.item():.4f}\n')
            run[0] += 1
            return loss
        optimizer.step(closure)

    output_img.data.clamp_(0,1)
    # return output_img

    # Save the image
    unloader = transforms.ToPILImage()
    image = output_img.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(output_img_path)
    print('Style transfer completed!')
    return True



