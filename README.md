# GhibliFilterTool

Welcome to **GhibliFilterTool**, an open-source project that harnesses Stable Diffusion and LoRA to bring the whimsical, hand-painted charm of Studio Ghibli to your images. Trained on an NVIDIA A100 GPU in Google Colab Pro, this tool is ready for you to explore, enhance, and collaborate on. I’m open-sourcing this to the GitHub community—fork it, improve it, and let’s take it further together. Cheers to the AI open-source community! 🎉

## 🌟 Features

- **Ghibli Magic**: Turn any photo into a vibrant, pastel-colored masterpiece with lush backgrounds and whimsical details.
- **Flexible**: Tweak strength, guidance scale, and prompts for custom results.
- **Community-Driven**: Open for collaboration and improvement.
- **Colab-Friendly**: Easy to run in Google Colab with GPU support.

---

## 🎨 Examples

See GhibliFilterTool in action:

| Before | After |
|--------|-------|
|![before1](https://github.com/user-attachments/assets/5a14c4d2-7ce0-477d-94f9-be8e55253a48) | ![after1](https://github.com/user-attachments/assets/99a5c7c0-f6c1-4f1d-ba11-ace3a6e55e8c) |

*Replace these placeholders with your before-and-after screenshots!*

---

## 🚀 Getting Started

### Running in Google Colab

Run GhibliFilterTool in Google Colab with a GPU (A100 recommended, but T4/V100 work too). Here’s how:

1. **Open Colab**:
   - Head to [Google Colab](https://colab.research.google.com/) and start a new notebook.

2. **Set GPU Runtime**:
   - Go to `Runtime` > `Change runtime type` > Select `GPU` > Save.

3. **Clone the Repo**:
   ```bash
   !git clone https://github.com/[YourUsername]/GhibliFilterTool.git
   %cd GhibliFilterTool
Install Dependencies:
bash

Collapse

Wrap

Copy
!pip install torch torchvision diffusers peft python-dotenv pillow
Upload Datasets:
Upload ghibli_dataset (1022 Ghibli images) and regularization_dataset (250 generic photos) via Colab’s file uploader or a cloud drive link.
Run the Code:
Copy the notebook cells from GhibliFilterTool.ipynb (or use snippets below) and execute them step-by-step.
Generate Art:
Upload an image in the final cell to see the Ghibli transformation!
🤝 Contributing
I’m open to collaboration and excited to see where the community takes this! Here’s how you can jump in:

Fork & Enhance: Fork the repo, tweak the model (e.g., more epochs, LoRA settings), and submit a pull request.
Improve the Model: Experiment with dataset size, hyperparameters, or inference options to perfect the Ghibli style.
Share Ideas: Open an issue for feature suggestions, bug reports, or discussions.
I trained this on an A100 GPU in Colab Pro, but you’re welcome to adapt it to other setups. Let’s make this a collaborative masterpiece—cheers to open-source AI! 🌍

📖 Technical Details
Overview
GhibliFilterTool fine-tunes Stable Diffusion v1.5 with LoRA to apply Studio Ghibli’s aesthetic. It uses 1022 Ghibli images for style and 250 regularization images for balance.

Current Model
Training Setup:
Hardware: NVIDIA A100 GPU (Colab Pro)
Dataset: 1022 Ghibli images, 250 regularization images
Epochs: 15 (early stopping at 6)
Batch Size: 4 (effective 8 with gradient accumulation)
LoRA: r=48, lora_alpha=96, target_modules=["to_q", "to_k", "to_v"]
LR: 5e-5, step decay every 7 epochs
Instance Weight: 2.0
Latest Training Log:
text

Collapse

Wrap

Copy
Starting fine-tuning with 1022 instance images and 250 regularization images...
Epoch 1/15 completed. Avg Instance Loss: 0.1402, Avg Reg Loss: 0.1572, LR: 0.000030
Epoch 2/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1520, LR: 0.000030
Epoch 3/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1507, LR: 0.000030
Epoch 4/15 completed. Avg Instance Loss: 0.1405, Avg Reg Loss: 0.1583, LR: 0.000030
Epoch 5/15 completed. Avg Instance Loss: 0.1462, Avg Reg Loss: 0.1515, LR: 0.000015
Epoch 6/15 completed. Avg Instance Loss: 0.1484, Avg Reg Loss: 0.1569, LR: 0.000015
Early stopping triggered after 6 epochs.
Performance: Instance loss ~0.14 (target <0.10). Output is decent but needs stronger Ghibli style.
Code Snippets
Training Setup
python

Collapse

Wrap

Copy
lora_config = LoraConfig(
    r=48,
    lora_alpha=96,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
Inference
python

Collapse

Wrap

Copy
def apply_ghibli_filter(input_path, strength=0.7, prompt="Studio Ghibli style artwork, vibrant pastel colors, whimsical characters, lush detailed backgrounds, hand-painted aesthetic"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.load_state_dict(torch.load(LORA_WEIGHTS_PATH, map_location=device), strict=False)
    init_image = Image.open(input_path).resize((512, 512))
    with torch.amp.autocast('cuda'):
        result = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=8.5, num_inference_steps=50).images[0]
    return result
Improvement Ideas
Boost LoRA: Try r=64, lora_alpha=128 on high-VRAM GPUs.
More Epochs: Extend to 20–25 with patience=10.
Dataset: Add more Ghibli images (2000+) or curate for quality.
Tuning: Test lr=1e-4, instance_weight=3.0, or larger effective batch sizes.
Inference: Experiment with strength=0.8–1.0, guidance_scale=9.0–10.0.
📜 License
MIT License—use, modify, and share freely!

🙌 Acknowledgements
AI Community: For tools and inspiration.
Hugging Face: For Stable Diffusion and diffusers.
Studio Ghibli: For the art that sparked this journey.
Let’s collaborate and bring more Ghibli magic to life! ✨

text

Collapse

Wrap

Copy

---

### Notes
- **Syntax**: This uses standard GitHub Markdown (`#` for headings, `-` for lists, ` ``` ` for code blocks, `|` for tables, etc.).
- **Placeholders**: Replace the image URLs in the "Examples" section with your actual screenshot paths (e.g., `![Before](images/before.jpg)` after uploading to your repo).
- **Username**: Update `[YourUsername]` with your GitHub handle.
- **File**: Save this as `README.md` in your repo root.

This README is GitHub-ready—upload it, add your screenshots, and watch the community engage! Let me know if you need further adjustments!






give the entire above as a markdown convention. entire above info 




Below is the entire content from my previous response formatted in proper GitHub Markdown (.md) syntax. This includes the corrected README content with all sections, headings, lists, code blocks, tables, and image placeholders formatted correctly for GitHub rendering.

markdown

Collapse

Wrap

Copy
Below is the corrected README in proper `.md` (Markdown) syntax for GitHub, based on the content I previously provided. I’ve ensured all headings, lists, code blocks, images, and tables use the correct Markdown conventions. This version is ready to be saved as `README.md` in your GitHub repository.

```markdown
# GhibliFilterTool

![GhibliFilterTool Banner](https://via.placeholder.com/1200x300.png?text=GhibliFilterTool+-+Transform+Your+Images+into+Studio+Ghibli+Art)  
*Transform your photos into Studio Ghibli-style artwork with AI!*

Welcome to **GhibliFilterTool**, an open-source project that harnesses Stable Diffusion and LoRA to bring the whimsical, hand-painted charm of Studio Ghibli to your images. Trained on an NVIDIA A100 GPU in Google Colab Pro, this tool is ready for you to explore, enhance, and collaborate on. I’m open-sourcing this to the GitHub community—fork it, improve it, and let’s take it further together. Cheers to the AI open-source community! 🎉

---

## 🌟 Features

- **Ghibli Magic**: Turn any photo into a vibrant, pastel-colored masterpiece with lush backgrounds and whimsical details.
- **Flexible**: Tweak strength, guidance scale, and prompts for custom results.
- **Community-Driven**: Open for collaboration and improvement.
- **Colab-Friendly**: Easy to run in Google Colab with GPU support.

---

## 🎨 Examples

See GhibliFilterTool in action:

| Before | After |
|--------|-------|
| ![Before Image](https://via.placeholder.com/300x300.png?text=Before+Image) | ![After Image](https://via.placeholder.com/300x300.png?text=After+Image+Ghibli+Style) |

*Replace these placeholders with your before-and-after screenshots!*

---

## 🚀 Getting Started

### Running in Google Colab

Run GhibliFilterTool in Google Colab with a GPU (A100 recommended, but T4/V100 work too). Here’s how:

1. **Open Colab**:
   - Head to [Google Colab](https://colab.research.google.com/) and start a new notebook.

2. **Set GPU Runtime**:
   - Go to `Runtime` > `Change runtime type` > Select `GPU` > Save.

3. **Clone the Repo**:
   ```bash
   !git clone https://github.com/[YourUsername]/GhibliFilterTool.git
   %cd GhibliFilterTool
Install Dependencies:
bash

Collapse

Wrap

Copy
!pip install torch torchvision diffusers peft python-dotenv pillow
Upload Datasets:
Upload ghibli_dataset (1022 Ghibli images) and regularization_dataset (250 generic photos) via Colab’s file uploader or a cloud drive link.
Run the Code:
Copy the notebook cells from GhibliFilterTool.ipynb (or use snippets below) and execute them step-by-step.
Generate Art:
Upload an image in the final cell to see the Ghibli transformation!
🤝 Contributing
I’m open to collaboration and excited to see where the community takes this! Here’s how you can jump in:

Fork & Enhance: Fork the repo, tweak the model (e.g., more epochs, LoRA settings), and submit a pull request.
Improve the Model: Experiment with dataset size, hyperparameters, or inference options to perfect the Ghibli style.
Share Ideas: Open an issue for feature suggestions, bug reports, or discussions.
I trained this on an A100 GPU in Colab Pro, but you’re welcome to adapt it to other setups. Let’s make this a collaborative masterpiece—cheers to open-source AI! 🌍

📖 Technical Details
Overview
GhibliFilterTool fine-tunes Stable Diffusion v1.5 with LoRA to apply Studio Ghibli’s aesthetic. It uses 1022 Ghibli images for style and 250 regularization images for balance.

Current Model
Training Setup:
Hardware: NVIDIA A100 GPU (Colab Pro)
Dataset: 1022 Ghibli images, 250 regularization images
Epochs: 15 (early stopping at 6)
Batch Size: 4 (effective 8 with gradient accumulation)
LoRA: r=48, lora_alpha=96, target_modules=["to_q", "to_k", "to_v"]
LR: 5e-5, step decay every 7 epochs
Instance Weight: 2.0
Latest Training Log:
text

Collapse

Wrap

Copy
Starting fine-tuning with 1022 instance images and 250 regularization images...
Epoch 1/15 completed. Avg Instance Loss: 0.1402, Avg Reg Loss: 0.1572, LR: 0.000030
Epoch 2/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1520, LR: 0.000030
Epoch 3/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1507, LR: 0.000030
Epoch 4/15 completed. Avg Instance Loss: 0.1405, Avg Reg Loss: 0.1583, LR: 0.000030
Epoch 5/15 completed. Avg Instance Loss: 0.1462, Avg Reg Loss: 0.1515, LR: 0.000015
Epoch 6/15 completed. Avg Instance Loss: 0.1484, Avg Reg Loss: 0.1569, LR: 0.000015
Early stopping triggered after 6 epochs.
Performance: Instance loss ~0.14 (target <0.10). Output is decent but needs stronger Ghibli style.
Code Snippets
Training Setup
python

Collapse

Wrap

Copy
lora_config = LoraConfig(
    r=48,
    lora_alpha=96,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
Inference
python

Collapse

Wrap

Copy
def apply_ghibli_filter(input_path, strength=0.7, prompt="Studio Ghibli style artwork, vibrant pastel colors, whimsical characters, lush detailed backgrounds, hand-painted aesthetic"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.load_state_dict(torch.load(LORA_WEIGHTS_PATH, map_location=device), strict=False)
    init_image = Image.open(input_path).resize((512, 512))
    with torch.amp.autocast('cuda'):
        result = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=8.5, num_inference_steps=50).images[0]
    return result
Improvement Ideas
Boost LoRA: Try r=64, lora_alpha=128 on high-VRAM GPUs.
More Epochs: Extend to 20–25 with patience=10.
Dataset: Add more Ghibli images (2000+) or curate for quality.
Tuning: Test lr=1e-4, instance_weight=3.0, or larger effective batch sizes.
Inference: Experiment with strength=0.8–1.0, guidance_scale=9.0–10.0.
📜 License
MIT License—use, modify, and share freely!

🙌 Acknowledgements
AI Community: For tools and inspiration.
Hugging Face: For Stable Diffusion and diffusers.
Studio Ghibli: For the art that sparked this journey.
Let’s collaborate and bring more Ghibli magic to life! ✨
