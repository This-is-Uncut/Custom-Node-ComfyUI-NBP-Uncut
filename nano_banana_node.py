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
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_ip": "127.0.0.1"}

def save_config(ip_address):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"last_ip": ip_address}, f)
    except Exception as e:
        print(f"Warning: Could not save Uncut NBP config: {e}")

initial_config = load_config()

def comfy_tensor_to_pil(tensor):
    """Converts ComfyUI tensor (B,H,W,C) to list of PIL Images."""
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
    ComfyUI Node for 'NBP'.
    Focused on 'gemini-3-pro-image-preview' -(Nano Banana Pro).
    Supports Reference Images + Extended Aspect Ratios + 4K.
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
                "aspect_ratio": (
                    [
                        "1:1",
                        "2:3", "3:2", 
                        "4:3", "3:4", 
                        "4:5", "5:4",
                        "9:16", "16:9", 
                        "21:9"
                    ],
                    {"default": "1:1"}
                ),
                "resolution": (
                    ["1K", "2K", "4K"], 
                    {"default": "1K"}
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # Supports Single Image OR Batch (stack of images)
                "reference_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate_nano"
    CATEGORY = "UncutNodes"

    def generate_nano(self, prompt, key_service_ip, aspect_ratio, resolution, seed, reference_images=None):
        
        # 1. Save IP and Fetch Key
        clean_ip = key_service_ip.strip()
        save_config(clean_ip)
        key_url = f"http://{clean_ip}:9191/get_gemini_key"
        print(f"--- Uncut NBP: Fetching key from {key_url} ---")
        
        try:
            response = requests.get(key_url, timeout=5)
            response.raise_for_status() 
            data = response.json()
            api_key = data.get("key")
            if not api_key: raise ValueError("Service returned JSON, but 'key' field was missing.")
        except Exception as e:
            raise ConnectionError(f"Failed to retrieve API Key. Have you entered the correct IP? Please contact the Uncut Admin. Error: {e}")

        # 2. Initialize Client
        client = genai.Client(api_key=api_key)
        
        # 3. Environment Check
        if not hasattr(types, "ImageConfig"):
            raise ImportError(
                "\n\nCRITICAL ERROR: 'google-genai' library is too old.\n"
                "Please inside your ComfyUI venv run: pip install --upgrade google-genai\n"
            )

        # 4. Process Images
        processed_images = []
        if reference_images is not None:
            processed_images = comfy_tensor_to_pil(reference_images)
            print(f"--- Detected {len(processed_images)} reference images in batch ---")

        print(f"--- Sending to Gemini 3 Pro (AR: {aspect_ratio}, Res: {resolution}) ---")

        try:
            # 5. Build Content List (Images first, then Prompt)
            contents = []
            if processed_images:
                contents.extend(processed_images)
            contents.append(prompt)

            # 6. API Call
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution 
                    ),
                    seed=seed % 2147483647 
                )
            )

            # 7. Process Output
            if hasattr(response, "parts"):
                for part in response.parts:
                    if part.text:
                        print(f"Gemini Reasoning: {part.text[:200]}...")
                    
                    try:
                        if hasattr(part, "as_image"):
                            pil_img = part.as_image()
                            if pil_img:
                                return (pil_to_comfy_tensor(pil_img),)
                    except Exception:
                        pass 

                    if part.inline_data:
                        d = part.inline_data.data
                        img_bytes = base64.b64decode(d) if isinstance(d, str) else d
                        return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)
            
            # Fallback for alternative structure
            elif hasattr(response, "candidates") and response.candidates:
                 for part in response.candidates[0].content.parts:
                    if part.text: print(f"Reasoning: {part.text[:100]}")
                    if part.inline_data:
                        d = part.inline_data.data
                        img_bytes = base64.b64decode(d) if isinstance(d, str) else d
                        return (pil_to_comfy_tensor(Image.open(io.BytesIO(img_bytes))),)

            raise Exception("Model returned success but no image was found in the response.")

        except Exception as e:
            print(f"!!!! Uncut NBP Error !!!!\n{e}\n")
            raise e

NODE_CLASS_MAPPINGS = {
    "NanoBananaProNode": NanoBananaProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaProNode": "Uncut NBP"
}