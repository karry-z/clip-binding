from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from . import get_model
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler


class DiffusionModel(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        # VAE
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Text Encoder
        text_config = deepcopy(config)
        text_config.model_name = config.text_model_name
        text_config.model_path = config.text_model_path
        self.text_encoder, _ = get_model(None, text_config, device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

        # Scheduler
        self.scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.vae = self.vae.to(device).to(dtype)
        self.text_encoder = self.text_encoder.to(device)
        self.unet = self.unet.to(device).to(dtype)
        
        self.height = 256
        self.width = 256
        self.num_inference_steps = 25
        self.guidance_scale = 7.5


    def forward(self, text, clean_images=None):
        if clean_images is None:
            return self.forward_infer(text)
        else:
            return self.forward_train(clean_images, text)

    def forward_infer(self, text):
        # get text embeddings
        text_embeddings = self.text_encoder.compute_text_representations(text)
        batch_size = len(text[0])
        cxt_size = len(text)
        uncond_embeddings = self.text_encoder.compute_text_representations([[""] * batch_size]*cxt_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.view(-1, 1, 768)
        

        # get noisy latents
        latents = torch.randn(
            (batch_size*cxt_size, self.unet.in_channels, self.height // 8, self.width // 8),
            generator=torch.manual_seed(0),
        )
        latents = latents.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        # denoise
        from tqdm.auto import tqdm

        self.scheduler.set_timesteps(self.num_inference_steps)


        for t in tqdm(self.scheduler.timesteps, disable=None):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            latent_model_input = latent_model_input.to(self.unet.dtype)
            text_embeddings = text_embeddings.to(self.unet.dtype)
            

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # get image
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents).sample
        return image
    

    def forward_train(self, clean_images, text):
        r'''
        input: img, text
        output: loss
        '''
        with torch.no_grad():
            # get clean latents
            clean_latents = self.vae.encode(clean_images.to(self.vae.dtype)).latent_dist.sample()

            # Sample noise to add to the images
            noise = torch.randn(clean_latents.shape).to(clean_latents.device)
            bs = clean_latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_latents = self.scheduler.add_noise(clean_latents, noise, timesteps)
            noisy_latents = noisy_latents.to(clean_latents.dtype)
            # get text embeddings
            if isinstance(text, str):
                text = [[text]]
            text_embeddings = self.text_encoder.compute_text_representations(text)
            text_embeddings = text_embeddings.view(-1, 1, 768).to(noisy_latents.dtype)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        loss = F.mse_loss(noise_pred, noise)
        return loss



def get_model_(config, device):
    model = DiffusionModel(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    if config.model_path is not None:
        model.load_state_dict(torch.load(config.model_path), strict=False)
    model = model.to(device)
    return model, optimizer


