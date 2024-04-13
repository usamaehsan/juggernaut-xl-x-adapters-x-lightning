



from typing import Optional
import torch
import os
from typing import List
import numpy as np
from PIL import Image
import cv2
import time
import sys

from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter, StableDiffusionXLAdapterPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    # KarrasDPM
)
from controlnet_aux import (
    HEDdetector,
    OpenposeDetector,
    MLSDdetector,
    CannyDetector,
)
from controlnet_aux.midas import MidasDetector
from controlnet_aux.lineart import LineartDetector

from compel import Compel
from diffusers.models import AutoencoderKL
import json
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
from diffusers.utils import load_image, make_image_grid


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    # "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

def resize_image(image, max_width, max_height):
    """
    Resize an image to a specific height while maintaining the aspect ratio and ensuring
    that neither width nor height exceed the specified maximum values.

    Args:
        image (PIL.Image.Image): The input image.
        max_width (int): The maximum allowable width for the resized image.
        max_height (int): The maximum allowable height for the resized image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get the original image dimensions
    original_width, original_height = image.size

    # Calculate the new dimensions to maintain the aspect ratio and not exceed the maximum values
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height

    # Choose the smallest ratio to ensure that neither width nor height exceeds the maximum
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(original_width * resize_ratio)
    new_height = int(original_height * resize_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def sort_dict_by_string(input_string, your_dict):
    if not input_string or not isinstance(input_string, str):
        # Return the original dictionary if the string is empty or not a string
        return your_dict

    order_list = [item.strip() for item in input_string.split(',')]

    # Include keys from the input string that are present in the dictionary
    valid_keys = [key for key in order_list if key in your_dict]

    # Include keys from the dictionary that are not in the input string
    remaining_keys = [key for key in your_dict if key not in valid_keys]

    sorted_dict = {key: your_dict[key] for key in valid_keys}
    sorted_dict.update({key: your_dict[key] for key in remaining_keys})

    return sorted_dict



class Predictor(BasePredictor):
    def setup(self):
        model_id = 'limiteinductive/Juggernaut-XL_v9_RunDiffusionPhoto_v2'

        self.vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        print("loading adapters")
        self.lineart_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
        ).to("cuda")

        self.depth_midas_adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, varient="fp16"
        ).to("cuda")

        print("loading detectors")
        self.midas_depth_detector = MidasDetector.from_pretrained( "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large").to("cuda")
        self.lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        print("loading pipe")
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=self.vae,
            torch_dtype=torch.float16,
            adapter=self.lineart_adapter
        ).to("cuda")

        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        self.pipe.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"),
                                     adapter_name="light"
                                     )
        self.pipe.load_lora_weights("ntc-ai/SDXL-LoRA-slider.micro-details-fine-details-detailed", weight_name='micro details, fine details, detailed.safetensors', adapter_name="micro-details")

        self.compel = Compel(tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2] , text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt - using compel, use +++ to increase words weight:: doc: https://github.com/damian0815/compel/tree/main/doc || https://invoke-ai.github.io/InvokeAI/features/PROMPTS/#attention-weighting",),
        negative_prompt: str = Input(
            description="Negative prompt - using compel, use +++ to increase words weight//// negative-embeddings available ///// FastNegativeV2 , boring_e621_v4 , verybadimagenegative_v1 || to use them, write their keyword in negative prompt",
            default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        ),
        num_inference_steps: int = Input(description="Steps to run denoising", default=20),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.0,
            ge=0.1,
            le=30.0,
        ),
        seed: int = Input(description="Seed", default=None),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        # num_outputs: int = Input(
        #     description="Number of images to generate",
        #     ge=1,
        #     le=10,
        #     default=1,
        # ),
        max_width: int = Input(
            description="Max width/Resolution of image",
            default=512,
        ),
        max_height: int = Input(
            description="Max height/Resolution of image",
            default=512,
        ),
        # consistency_decoder: bool = Input(
        #     description="Enable consistency decoder",
        #     default=True,
        # ),
        scheduler: str = Input(
            default="DDIM",
            choices=SCHEDULERS.keys(),
            description="Choose a scheduler.",
        ),
        lineart_image: Path = Input(
            description="Control image for lineart adapter", default=None
        ),
        lineart_conditioning_scale: float = Input(
            description="Conditioning scale for canny controlnet",
            default=1,
        ),
        depth_image: Path = Input(
            description="Control image for depth controlnet", default=None
        ),
        depth_conditioning_scale: float = Input(
            description="Conditioning scale for depth controlnet",
            default=1,
        ),
        # inpainting_image: Path = Input(
        #     description="Control image for inpainting controlnet", default=None
        # ),
        # mask_image: Path = Input(
        #     description="mask image for inpainting controlnet", default=None
        # ),
        # positive_auto_mask_text: str = Input(
        #     description="// seperated list of objects for mask, AI will auto create mask of these objects, if mask text is given, mask image will not work - 'hairs // eyes // cloth'", default=None
        # ),
        # negative_auto_mask_text: str = Input(
        #     description="// seperated list of objects you dont want to mask - 'hairs // eyes // cloth' ", default=None
        # ),
        # inpainting_conditioning_scale: float = Input(
        #     description="Conditioning scale for inpaint controlnet",
        #     default=1,
        # ),
        # inpainting_strength: float = Input(
        #     description="inpainting strength",
        #     default=1,
        # ),
        # sorted_controlnets: str = Input(
        #     description="Comma seperated string of controlnet names, list of names: tile, inpainting, lineart,depth ,scribble , brightness /// example value: tile, inpainting, lineart ", default="lineart, tile, inpainting"
        # ),
        ip_adapter_image: Path = Input(
            description="IP Adapter image", default=None
        ),
        ip_adapter_weight: float = Input(
            description="IP Adapter weight", default=1.0, 
        ),
        lightning_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ),
        micro_detail_lora_weight: float = Input(
            description="disabled on 0", default=0,
        ), 
    ) -> List[Path]:
        print("1")
        if not disable_safety_check and 'nude' in prompt:
            raise Exception(
                f"NSFW content detected. try a different prompt."
            )
        
        adapters = []
        conditioning_scales= []
        images = []
        w = None
        h= None
        print("2")
        if lineart_image:
            adapters.append(self.lineart_adapter)
            conditioning_scales.append(lineart_conditioning_scale)
            print("3")
            lineart_image = Image.open(lineart_image).convert('RGB')
            img = self.lineart_detector(lineart_image, detect_resolution=384, image_resolution=1024).convert('RGB')
            print("4")
            i = resize_image(img, max_width if not w else w, max_height if not w else h)
            if not w:
                w, h= i.size
            images.append(i)
            print("5")
        
        if depth_image:
            depth_image = Image.open(depth_image).convert('RGB')
            adapters.append(self.depth_midas_adapter)
            conditioning_scales.append(depth_conditioning_scale)
            img = self.midas_depth_detector(depth_image, detect_resolution=384, image_resolution=1024).convert('RGB')
            i = resize_image(img, max_width if not w else w, max_height if not w else h)
            if not w:
                w, h= i.size
            images.append(i)
        
        print("6")
        lora_weights=[]
        loras= []
        if lightning_lora_weight!=0:
            lora_weights.append(lightning_lora_weight)
            loras.append("light")

        if micro_detail_lora_weight!=0:
            lora_weights.append(micro_detail_lora_weight)
            loras.append("micro_details")
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print("7")
        if ip_adapter_image:
            self.pipe.set_ip_adapter_scale(ip_adapter_weight)
            ip_image = resize_image(ip_adapter_image)
        else:
            self.pipe.set_ip_adapter_scale(0)
            ip_image = resize_image("u.jpg")
        
        self.pipe.set_adapters(loras, adapter_weights=lora_weights)
        self.pipe.fuse_lora()
        self.pipe.adapter = adapters


        if disable_safety_check:
            self.pipe.safety_checker = None

        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)
        print("8")
        conditioning, pooled = self.compel(prompt)
        n_conditioning, n_pooled = self.compel(negative_prompt)
        print("9")
        output = self.pipe(
            # prompt, negative_prompt= negative_prompt,
            prompt_embeds=conditioning, pooled_prompt_embeds=pooled,
            negative_prompt_embeds = n_conditioning, negative_pooled_prompt_embeds = n_pooled,
            image= images,
            adapter_conditioning_scale=conditioning_scales, 
            num_inference_steps= num_inference_steps,
            ip_adapter_image=ip_image,
            guidance_scale= guidance_scale
            )
        outputs= [output]

        self.pipe.unfuse_lora()

        output_paths= []
        i=0
        for output in outputs:
            output_path = f"/tmp/output_{i}.png"
            output.images[0].save(output_path)
            output_paths.append(Path(output_path))
            i+=1

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


