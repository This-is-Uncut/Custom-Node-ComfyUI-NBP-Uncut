import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
from google import genai
from google.genai import types

# --- Configuration Persistence Logic ---
CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "uncut_nbp_config.json")

def load_config():
    """Loads the saved IP address from config file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_ip": "127.0.0.1"}

def save_config(ip_address):
    """Saves the current IP address to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"last_ip": ip_address}, f)
    except Exception as e:
        print(f"Warning: Could not save Uncut NBP config: {e}")

# Load configuration on module import
initial_config = load_config()

def comfy_tensor_to_pil(tensor):
    if tensor is None:
        return []
    tensor = tensor.cpu().detach()
    pil_images = []
    for i in range(tensor.shape[0]):
        img_np = np.clip(tensor[i].numpy() * 255.0, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_np, 'RGB'))
    return pil_images

def pil_to_comfy_tensor(pil_image):
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0)
    return tensor

class NanoBananaProNode:
    """
    ComfyUI Node for 'Uncut NBP'.
    Fetches API Key from local network service (Port 9191).
    Fix: Injects Aspect Ratio into prompt for Gemini models to ensure consistency.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        current_config = load_config()
        current_ip = current_config.get("last_ip", "127.0.0.1")
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Generate a cinematic shot of a futuristic city with neon lights based on these references."
                }),
                "key_service_ip": ("STRING", {
                    "default": current_ip, 
                    "multiline": False
                }),
                "model_version": (
                    [
                        "gemini-3-pro-image-preview", 
                        "gemini-2.0-flash-exp", 
                        "imagen-3.0-generate-001",    
                    ], 
                    {"default": "gemini-3-pro-image-preview"} 
                ),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    {"default": "1:1"}
                ),
                "resolution": (
                    ["1k", "2k", "4k"],
                    {"default": "1k"}
                ),
                "safety_filter": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"], {"default": "BLOCK_NONE"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_nano"
    CATEGORY = "UncutNodes"

    def generate_nano(self, prompt, key_service_ip, model_version, aspect_ratio, resolution, safety_filter, seed, reference_images=None):
        
        # --- Persistence Step ---
        clean_ip = key_service_ip.strip()
        save_config(clean_ip)

        # --- 1. Fetch API Key ---
        key_url = f"http://{clean_ip}:9191/get_gemini_key"
        print(f"--- Uncut NBP: Fetching key from {key_url} ---")
        
        try:
            response = requests.get(key_url, timeout=5)
            response.raise_for_status() 
            data = response.json()
            api_key = data.get("key")
            
            if not api_key:
                raise ValueError(f"Service returned JSON, but 'key' field was missing.")
                
        except Exception as e:
            raise ConnectionError(f"Failed to retrieve API Key from {key_url}. Error: {e}")

        # --- 2. Configure Client ---
        client = genai.Client(api_key=api_key)
        api_seed = seed % 2147483647 

        safety_mapping = {
            "BLOCK_NONE": "BLOCK_NONE", 
            "BLOCK_ONLY_HIGH": "BLOCK_ONLY_HIGH", 
            "BLOCK_MEDIUM_AND_ABOVE": "BLOCK_MEDIUM_AND_ABOVE"
        }
        
        print(f"--- Uncut NBP Node: Using {model_version} ---")
        
        # Handle Images
        processed_images = []
        if reference_images is not None:
            processed_images = comfy_tensor_to_pil(reference_images)
            print(f"--- Detected {len(processed_images)} reference images in batch ---")

        try:
            # --- PATH A: Imagen Models (Strict Config) ---
            # Imagen prefers parameters over prompt instructions for AR.
            if "imagen" in model_version:
                print(f"Detected Imagen model (Using Config for AR).")
                
                # For Imagen, we just append quality to prompt, AR goes to config below
                imagen_prompt = f"{prompt} --quality {resolution}"

                response = client.models.generate_images(
                    model=model_version,
                    prompt=imagen_prompt,
                    config=types.GenerateImagesConfig(
                        aspect_ratio=aspect_ratio, # Strict API Parameter
                        number_of_images=1,
                        output_mime_type="image/png",
                        seed=api_seed,
                        safety_filter_level=safety_mapping[safety_filter],
                        person_generation="allow_adult"
                    )
                )
                if response.generated_images:
                    img_bytes = response.generated_images[0].image.image_bytes
                    return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)

            # --- PATH B: Gemini 3 / Nano Banana (Reasoning/Prompt Based) ---
            # Gemini acts like a chatbot. It often IGNORES config parameters for images 
            # if they aren't explicit in the conversation. We force it in the prompt.
            
            # Construct a robust prompt instruction
            gemini_prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Generate the image with Aspect Ratio {aspect_ratio}. "
                f"Target Quality: {resolution}."
            )

            contents = []
            contents.extend(processed_images)
            contents.append(gemini_prompt)

            print(f"Sending request to Gemini... (Prompt-forced AR: {aspect_ratio})")
            
            response = client.models.generate_content(
                model=model_version,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"], 
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_mapping[safety_filter]),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_mapping[safety_filter]),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_mapping[safety_filter]),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_mapping[safety_filter])
                    ],
                    seed=api_seed
                )
            )

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        print(f"Gemini Reasoning: {part.text[:100]}...")
                    if part.inline_data:
                        d = part.inline_data.data
                        img_bytes = base64.b64decode(d) if isinstance(d, str) else d
                        return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)
            
            raise Exception("Model finished but no image was found in the output.")

        except Exception as e:
            print(f"!!!! Uncut NBP Error !!!!\n{e}\n")
            raise e

NODE_CLASS_MAPPINGS = {
    "NanoBananaProNode": NanoBananaProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProNode": "Uncut NBP"
}
