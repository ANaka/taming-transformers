import argparse
import math
from pathlib import Path
import sys
from datetime import datetime
import os
import shutil
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import taming
import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
import json
import gc
from dataclasses import dataclass, field, asdict
import copy
from tqdm.notebook import tqdm
import signal
import fn

class GracefulExiter():
    # definitely jacked this from somewhere on stack overflow
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state

def default_field(obj):
    return field(default_factory=lambda: copy.deepcopy(obj))

_models_dir = Path(taming.models.__path__[0])
    

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
 
 
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply
 
class PromptStr(str):

  def __or__(self, other):
    
    return PromptStr(" | ".join([self, other]))

  def __getitem__(self, idx):
    return PromptStr(self + ":" + str(idx))

P = PromptStr
P("ayy")[3] | "lmao" | P("trending on artstationHQ")[.2]
 
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply
 
 
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
 
 
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
 
 
def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])
 

 
 
def _load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def load_vqgan_model(
    model_name='vqgan_imagenet_f16_16384',
    models_dir=None,
    ):
    if models_dir is None:
        models_dir = _models_dir
    models_dir = Path(models_dir)
    return _load_vqgan_model(
        config_path=models_dir.joinpath(f'{model_name}.yaml'),
        checkpoint_path=models_dir.joinpath(f'{model_name}.ckpt'))
    
def load_clip_model(clip_model='ViT-B/32'):
    return clip.load(clip_model, jit=False)[0].eval().requires_grad_(False)
 
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def empty_ram():
    for var in ['device', 'model', 'perceptor', 'z']:
        try:
            del globals()[var]
        except:
            pass
    try:
        gc.collect()
    except:
        pass

    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    
@dataclass
class LatentSpacewalkParameters(object):
    texts: list = field(default_factory=list)
    initial_image: str = None
    target_images:list = field(default_factory=list)
    seed: int = None
    max_iterations: int = None
    learning_rate: float = 0.2
    save_interval: int = 1
    zoom_interval: int = None
    display_interval: int = 3
    init_weight:int = 0
    init_from_last_saved_image:bool =True
    
    def __post_init__(self):
        self.texts = [phrase.strip() for phrase in self.texts]
        
    
    @property
    def prms(self):
        return asdict(self)
    
             
        
        
class Spacewalker(object):
    
    def __init__(
        self, 
        parameters,
        vqgan_model_name='vqgan_imagenet_f16_16384',
        cutn: int = 64,
        cut_pow: float = 1.,
        width: int = 640,
        height: int = 480,
        root_savedir: str = '/home/naka/art/latent_spacewalker',
        nft_id:str = None,
        models_dir='', # '/content/drive/MyDrive/vqgan_models'
        ):
        
        self.p = parameters
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.width = width
        self.height = height
        self.vqgan_model_name = vqgan_model_name
        self.models_dir = models_dir
        if nft_id is None:
            self.nft_id = fn.new_nft_id()
        self.root_savedir = Path(root_savedir)
        self.image_savedir = self.root_savedir.joinpath(f'{self.nft_id}')
        os.makedirs(self.image_savedir, exist_ok=True)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = load_vqgan_model(model_name=vqgan_model_name).to(self.device)
        self.perceptor = load_clip_model().to(self.device)
        
        
        cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, self.cutn, cut_pow=self.cut_pow)
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = self.width // f, self.height // f
        self.sideX, self.sideY = self.toksX * f, self.toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        
        
        
        self.ii = 0
        self.last_saved_filename = None
        
    @property
    def size(self):
        return (self.sideX, self.sideY)
    
    @property
    def png_metadata(self):
        metadata = PngInfo()
        gen_config = {
            "texts": self.p.texts,
            "width": self.width,
            "height": self.height,
            "init_image": self.p.initial_image,
            "target_images": self.p.target_images,
            "learning_rate": self.p.learning_rate,
            "training_seed": self.p.seed,
            "model": self.vqgan_model_name,
        }
        for k, v in gen_config.items():
            try:
                metadata.add_text("lsw_ " + k, str(v))
            except UnicodeEncodeError:
                pass
        return metadata
    
    @property
    def zoom_transforms(self):
        zoom_transforms = torch.nn.Sequential(
            transforms.CenterCrop((self.sideY-self.p.n_pixels_zoom, self.sideX-self.p.n_pixels_zoom)),
            transforms.Resize((self.sideY, self.sideX))
        )
#         scripted_transforms = torch.jit.script(zoom_transforms)
        return zoom_transforms.to(self.device)
    
    def initialize_z(self):
        if self.p.init_from_last_saved_image and (self.last_saved_filename is not None):
            img_to_load = self.last_saved_filename
        else:
            img_to_load = self.p.initial_image
            
        if img_to_load is not None:
            pil_image = Image.open(img_to_load).convert('RGB')
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            self.z_current = self.encode_PIL_image(pil_image)
            return self.z_current
        else:
            one_hot = F.one_hot(torch.randint(self.n_toks, [self.toksY * self.toksX], device=self.device), self.n_toks).float()
            z = one_hot @ self.model.quantize.embedding.weight
            self.z_current = z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)
            return self.z_current
        
        
    def set_z_orig(self):
        self.z_orig = self.z_current.clone()
        
    def encode_PIL_image(self, pil_image):
        z, *_ = self.model.encode(TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)  # is the adding/mult undoing rescaling from quantization?
        return z
    
    def encode_image_prompt(self, image_prompt):
        path, weight, stop = parse_prompt(image_prompt)
        img = resize_image(Image.open(path).convert('RGB'), (self.sideX, self.sideY))
        batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
        embed = self.perceptor.encode_image(self.normalize(batch)).float()
        return Prompt(embed, weight, stop).to(self.device)
    
    def encode_text(self, text):
        return self.perceptor.encode_text(clip.tokenize(text).to(self.device)).float()
        
    def encode_text_prompt(self, text_prompt):
        text, weight, stop = parse_prompt(text_prompt)
        embed = self.encode_text(text)
        return Prompt(embed, weight, stop).to(self.device)
        
    def encode_prompts(self):
        self.pMs = []
        for prompt in self.p.texts:
            self.pMs.append(self.encode_text_prompt(prompt))
            
        for prompt in self.p.target_images:
            self.pMs.append(self.encode_image_prompt(prompt))
            
            
    def initialize(self):
        self.initialize_z()
        self.z_current.requires_grad_(True)
        self.opt = optim.Adam([self.z_current], lr=self.p.learning_rate)
        self.encode_prompts()
        
            
    def synth(self, z):
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)
    
    @torch.no_grad()
    def checkin(self, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {self.ii}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z_current)
        TF.to_pil_image(out[0].cpu()).save('progress.png', pnginfo=self.png_metadata)
        display.display(display.Image('progress.png'))
    
    @property
    def longname(self):
        return ''.join([self.nft_id] + [s.replace(' ', '_') for s in self.p.texts])
    
    def ascend_txt(self):
        
        out = self.synth(self.z_current)
        
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.p.init_weight:
            result.append(F.mse_loss(self.z_current, self.z_orig) * self.p.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        if self.ii % self.p.save_interval == 0:
            img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
            img = np.transpose(img, (1, 2, 0))
            filename = self.image_savedir.joinpath(f'{self.longname}{self.ii:04}.png')
            imageio.imwrite(filename, np.array(img))
            self.last_saved_filename = filename.as_posix()
            self.t = out
            self.img = img
            
        return result
    
    @property
    def pil_img(self):
        return TF.to_pil_image(self.img.cpu())
    
    def zoom(self):
        with torch.no_grad():
            zoomed_output = self.zoom_transforms(self.synth(self.z_current)) * 2 - 1
            self.z_current, *_ = self.model.encode(zoomed_output)
            self.z_current.copy_(self.z_current.maximum(self.z_min).minimum(self.z_max))
        self.z_current.requires_grad_(True)
        self.opt = optim.Adam([self.z_current], lr=self.p.learning_rate)
        self.encode_prompts()
        
    def train(self):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        if self.ii % self.p.display_interval == 0:
            self.checkin(lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z_current.copy_(self.z_current.maximum(self.z_min).minimum(self.z_max))
            
                
    def run(self, parameters=None):
        if parameters is not None:
            self.p = parameters
        self.iter_to_stop_at = self.p.max_iterations + self.ii
        self.initialize()
        flag = GracefulExiter()
        while self.ii < self.iter_to_stop_at:
            if self.p.zoom_interval :
                if self.ii % self.p.zoom_interval == 0:
                    self.zoom()
            self.train()
            self.ii += 1
            if flag.exit():
                break
            
            
#https://github.com/nerdyrodent/VQGAN-CLIP/blob/main/generate.py         
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., augments=None):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        if augments is None:
            augments = [
                'Ji', 'Sh', 'Gn', 'Pe', 'Ro', 'Af', 'Et', 'Ts', 'Cr', 'Er', 'Re',
            ]
        
        # Pick your own augments & their order
        augment_list = []
        for item in self.augments:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.4, p=0.7))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=15, p=0.7, padding_mode='border', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.3, same_on_batch=False, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.05, .33), ratio=(.3, 1.3), same_on_batch=True, p=0.5))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
        
        # print(augment_list)
        
        self.augs = nn.Sequential(*augment_list)

        '''
        self.augs = nn.Sequential(
            # Original:
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5), 
            # Updated colab:
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),        
            )
        '''
            
        self.noise_fac = 0.1
        # self.noise_fac = False
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        # sideY, sideX = input.shape[2:4]
        # max_size = min(sideX, sideY)
        # min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)
            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
    
    
 
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch