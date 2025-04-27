# GhibliFilterTool

Welcome to **GhibliFilterTool**, an open-source project that harnesses Stable Diffusion and LoRA to bring the whimsical, hand-painted charm of Studio Ghibli to your images. Trained on an NVIDIA A100 GPU in Google Colab Pro, this tool is ready for you to explore, enhance, and collaborate on. I’m open-sourcing this to the GitHub community—fork it, improve it, and let’s take it further together. Cheers to the AI open-source community! 🎉

- I urge the open-source AI community to enhance and refine this model, ensuring AI remains accessible to everyone. To achieve this, we need greater GPU power, increased computing capacity, and more extensive training data.


## 🌟 Features

- **Ghibli Magic**: Turn any photo into a vibrant, pastel-colored masterpiece with lush backgrounds and whimsical details.
- **Flexible**: Tweak strength, guidance scale, and prompts for custom results.
- **Community-Driven**: Open for collaboration and improvement.
- **Colab-Friendly**: Easy to run in Google Colab with GPU support.

## 🎨 Examples

See GhibliFilterTool in action:

| Before | After |
|--------|-------|
|![before1](https://github.com/user-attachments/assets/5a14c4d2-7ce0-477d-94f9-be8e55253a48) | ![after1](https://github.com/user-attachments/assets/99a5c7c0-f6c1-4f1d-ba11-ace3a6e55e8c) |
|![before2](https://github.com/user-attachments/assets/90f301d0-96ed-4250-836b-73b371d5697d) | ![after2](https://github.com/user-attachments/assets/1a8e9208-12f3-417d-b71a-8d613b01728c) |
|![image](https://github.com/user-attachments/assets/df568d20-8825-422d-b6e6-f37463f06b12) | ![image](https://github.com/user-attachments/assets/ed095a7c-195a-41e7-a641-138f5e219645)

## Architecture

![diagram (1)](https://github.com/user-attachments/assets/0080e4e1-bceb-47ff-86f3-e7cbaaaddc2f)


## 🚀 Getting Started

## Drive Link to Datasets 

-https://drive.google.com/drive/folders/1mxNdEK88t2OSdE5b30eMws4e7HoXRsGr?usp=sharing

-https://drive.google.com/drive/folders/1VI_N3IorCHcrtf_QwmYVdlI1LT3kCa4X?usp=sharing

## Drive Link to already fine tuned model for the above datasets: 

-https://drive.google.com/file/d/13wbRL1hFrjUREBR2J-qW-HUWvecEDdzY/view?usp=sharing


### Running in Google Colab with A100 GPU

Run GhibliFilterTool in Google Colab with a GPU (A100 recommended, but T4/V100 work too). Here’s how:

1. Open Colab:
   
- Head to [Google Colab](https://colab.research.google.com/) and start a new notebook.

2. Set GPU Runtime:

- Go to `Runtime` > `Change runtime type` > Select `GPU A100` > Save.

3. Clone the Repo: git clone https://github.com/[YourUsername]/GhibliFilterTool.git
   
**Install Dependencies**: !pip install torch torchvision diffusers peft python-dotenv pillow

**Upload Datasets**: Upload ghibli_dataset (1022 Ghibli images) and regularization_dataset (1000 + generic photos) via Colab’s file uploader or a cloud drive link. (i've included the folders containing the images as required) 

**Run the Code**: Copy the notebook cells from GhibliFilterTool.ipynb (or use snippets below) and execute them step-by-step.

**Generate Art**: Upload an image in the final cell to see the Ghibli transformation!

## 🤝 Contributing

I’m open to collaboration and excited to see where the community takes this! Here’s how you can jump in:

## Fork & Enhance

**Fork the repo, tweak the model (e.g., more epochs, LoRA settings), and submit a pull request.**

## Improve the Model

**Experiment with dataset size, hyperparameters, or inference options to perfect the Ghibli style.**

## Share Ideas

-Open an issue for feature suggestions, bug reports, or discussions.

**I trained this on an A100 GPU in Colab Pro, but you’re welcome to adapt it to other setups. Let’s make this a collaborative masterpiece—cheers to open-source AI! 🌍**

## 📖 Technical Details Overview:

## Current Model

**GhibliFilterTool fine-tunes Stable Diffusion v1.5 with LoRA to apply Studio Ghibli’s aesthetic. It uses 1022 Ghibli images for style and 250 regularization images for balance.**

## Training Setup:

Hardware: NVIDIA A100 GPU (Colab Pro)
Dataset: 1022 Ghibli images, 250 regularization images
Epochs: 15 (early stopping at 6)
Batch Size: 4 (effective 8 with gradient accumulation)
LoRA: r=48, lora_alpha=96, target_modules=["to_q", "to_k", "to_v"]
LR: 5e-5, step decay every 7 epochs
Instance Weight: 2.0

## Latest Training Log:

Starting fine-tuning with 1022 instance images and 250 regularization images...
Epoch 1/15 completed. Avg Instance Loss: 0.1402, Avg Reg Loss: 0.1572, LR: 0.000030
Epoch 2/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1520, LR: 0.000030
Epoch 3/15 completed. Avg Instance Loss: 0.1446, Avg Reg Loss: 0.1507, LR: 0.000030
Epoch 4/15 completed. Avg Instance Loss: 0.1405, Avg Reg Loss: 0.1583, LR: 0.000030
Epoch 5/15 completed. Avg Instance Loss: 0.1462, Avg Reg Loss: 0.1515, LR: 0.000015
Epoch 6/15 completed. Avg Instance Loss: 0.1484, Avg Reg Loss: 0.1569, LR: 0.000015
Early stopping triggered after 6 epochs.
Performance: Instance loss ~0.14 (target <0.10). Output is decent but needs stronger Ghibli style.

## Improvement Ideas:

-Boost LoRA: Try r=64, lora_alpha=128 on high-VRAM GPUs.
-More Epochs: Extend to 20–25 with patience=10.
-Dataset: Add more Ghibli images (2000+) or curate for quality.
-Tuning: Test lr=1e-4, instance_weight=3.0, or larger effective batch sizes.
-Inference: Experiment with strength=0.8–1.0, guidance_scale=9.0–10.0.

## 📜 License

**MIT License—use, modify, and share freely!**

## 🙌 Acknowledgements
**AI Community**: For tools and inspiration.
**Hugging Face**: For Stable Diffusion and diffusers.
**Studio Ghibli**: For the art that sparked this journey and my personal interest in Anime!
**Let’s collaborate and bring more Ghibli magic to life! ✨**
