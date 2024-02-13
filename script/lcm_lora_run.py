import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

from PIL import Image
import datetime
import os

model_id = "Lykon/dreamshaper-7"
adapter_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()


# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
prompt = "Potato and shrimp gratin, 1k"


# disable guidance_scale by passing 0
image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]



# Convert the tensor to a PIL Image
image_pil = Image.fromarray((image.clamp(0, 1).mul(255).byte().cpu().numpy().transpose(1, 2, 0)))

# Create directory if it does not exist
os.makedirs('/root/autodl-tmp/pic/', exist_ok=True)

# Format the filename with the prompt and current timestamp
filename = f"/root/autodl-tmp/pic/{prompt.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

# Save the image
image_pil.save(filename)
