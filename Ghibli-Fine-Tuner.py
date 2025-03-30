# Cell 1: Install Dependencies
!pip install torch torchvision diffusers peft python-dotenv pillow

# Cell 2: Import Libraries and Setup
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from IPython.display import display
import google.colab.files
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"VRAM Available: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Cell 3: Define Environment Variables
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "ghibli_lora_weights.pt"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
DATASET_DIR = "ghibli_dataset"
REGULARIZATION_DIR = "regularization_dataset"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(REGULARIZATION_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Directories created:", DATASET_DIR, REGULARIZATION_DIR, INPUT_DIR, OUTPUT_DIR)

# Cell 4: Upload Ghibli Dataset Folder
print("Upload your 'ghibli_dataset' folder containing 1022 Ghibli images (select all images inside the folder):")
uploaded = google.colab.files.upload()
if uploaded:
    for filename, content in uploaded.items():
        if filename.endswith((".jpg", ".png")):
            target_path = os.path.join(DATASET_DIR, filename)
            with open(target_path, "wb") as f:
                f.write(content)
    image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith((".jpg", ".png"))]
    print(f"Uploaded {len(image_files)} image files to {DATASET_DIR}: {image_files[:5]} (showing first 5)")
    if len(image_files) == 0:
        raise Exception("No valid image files uploaded to ghibli_dataset! Ensure you selected .jpg or .png files.")
else:
    raise Exception("No files uploaded! Please upload your ghibli_dataset folder contents.")

# Cell 5: Upload Regularization Dataset Folder (Capped at 250)
print("Upload your 'regularization_dataset' folder containing up to 250 generic photos (select images inside the folder):")
uploaded = google.colab.files.upload()
if uploaded:
    reg_count = 0
    for filename, content in uploaded.items():
        if filename.endswith((".jpg", ".png")) and reg_count < 250:
            target_path = os.path.join(REGULARIZATION_DIR, filename)
            with open(target_path, "wb") as f:
                f.write(content)
            reg_count += 1
    reg_files = [f for f in os.listdir(REGULARIZATION_DIR) if f.endswith((".jpg", ".png"))]
    print(f"Uploaded {len(reg_files)} image files to {REGULARIZATION_DIR}: {reg_files[:5]} (showing first 5)")
    if len(reg_files) == 0:
        raise Exception("No valid image files uploaded to regularization_dataset! Ensure you selected .jpg or .png files.")
else:
    print("No regularization files uploaded. Proceeding without regularization.")

# Cell 6: Define Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset_dir, is_instance=True):
        self.image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png"))]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.is_instance = is_instance

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), "Studio Ghibli style" if self.is_instance else "a photo"

# Cell 7: Fine-Tuning Function
def fine_tune_model():
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to(device)

    lora_config = LoraConfig(
        r=48,
        lora_alpha=96,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.1
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    instance_dataset = CustomDataset(DATASET_DIR, is_instance=True)
    reg_dataset = CustomDataset(REGULARIZATION_DIR, is_instance=False)
    if len(instance_dataset) == 0:
        raise Exception("No images found in ghibli_dataset!")
    if len(reg_dataset) == 0:
        print("Warning: No regularization images. Training without regularization.")
    instance_loader = DataLoader(instance_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    reg_loader = DataLoader(reg_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True) if reg_dataset else None
    reg_iterator = iter(reg_loader) if reg_loader else None

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    pipe.unet.train()
    num_epochs = 15
    best_loss = float('inf')
    patience = 7
    no_improve = 0
    instance_weight = 2.0
    accum_steps = 2

    print(f"Starting fine-tuning with {len(instance_dataset)} instance images and {len(reg_dataset)} regularization images...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for i, instance_batch in enumerate(instance_loader):
            imgs, prompt = instance_batch
            batch_size = imgs.shape[0]
            imgs = imgs.to(device, dtype=torch.float16)

            with torch.no_grad():
                latents = pipe.vae.encode(imgs).latent_dist.sample() * pipe.vae.config.scaling_factor

            text_inputs = pipe.tokenizer([prompt[0]] * batch_size, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            encoder_hidden_states = pipe.text_encoder(**text_inputs)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast('cuda'):
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = instance_weight * torch.nn.functional.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() / instance_weight
            num_batches += 1

            if reg_iterator:
                try:
                    reg_batch = next(reg_iterator)
                except StopIteration:
                    reg_iterator = iter(reg_loader)
                    reg_batch = next(reg_iterator)
                reg_imgs, reg_prompt = reg_batch
                reg_batch_size = reg_imgs.shape[0]
                reg_imgs = reg_imgs.to(device, dtype=torch.float16)
                with torch.no_grad():
                    reg_latents = pipe.vae.encode(reg_imgs).latent_dist.sample() * pipe.vae.config.scaling_factor
                reg_text_inputs = pipe.tokenizer([reg_prompt[0]] * reg_batch_size, return_tensors="pt", padding=True)
                reg_text_inputs = {k: v.to(device) for k, v in reg_text_inputs.items()}
                reg_encoder_hidden_states = pipe.text_encoder(**reg_text_inputs)[0]
                reg_noise = torch.randn_like(reg_latents)
                reg_timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (reg_batch_size,), device=device).long()
                reg_noisy_latents = pipe.scheduler.add_noise(reg_latents, reg_noise, reg_timesteps)
                with torch.amp.autocast('cuda'):
                    reg_noise_pred = pipe.unet(noisy_latents, reg_timesteps, reg_encoder_hidden_states).sample
                    reg_loss = torch.nn.functional.mse_loss(reg_noise_pred, reg_noise)
                scaler.scale(reg_loss).backward()

                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_reg_loss += reg_loss.item()

        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches if reg_iterator else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Avg Instance Loss: {avg_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    torch.save(pipe.unet.state_dict(), LORA_WEIGHTS_PATH)
    print(f"Fine-tuned weights saved to {LORA_WEIGHTS_PATH}")

# Cell 8: Run Training
fine_tune_model()

# Cell 9: Inference Function
def apply_ghibli_filter(input_path, strength=0.7, prompt="Studio Ghibli style artwork, vibrant pastel colors, whimsical characters, lush detailed backgrounds, hand-painted aesthetic"):
    print("Loading inference pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    lora_config = LoraConfig(
        r=48,
        lora_alpha=96,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.1
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.load_state_dict(torch.load(LORA_WEIGHTS_PATH, map_location=device), strict=False)

    init_image = Image.open(input_path).resize((512, 512))
    with torch.amp.autocast('cuda'):
        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=8.5,
            num_inference_steps=50
        ).images[0]
    return result

# Cell 10: Upload and Generate
print("Upload an image to transform:")
uploaded = google.colab.files.upload()
if uploaded:
    input_filename = list(uploaded.keys())[0]
    input_path = os.path.join(INPUT_DIR, input_filename)
    with open(input_path, "wb") as f:
        f.write(uploaded[input_filename])
    
    output_path = os.path.join(OUTPUT_DIR, f"ghibli_{input_filename}")
    result = apply_ghibli_filter(input_path)
    result.save(output_path)
    display(result)
    print(f"Generated image saved to {output_path}")
else:
    print("No image uploaded!")

# Cell 11: Download Weights (Optional)
google.colab.files.download(LORA_WEIGHTS_PATH)
