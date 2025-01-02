from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionConfig
from modeling_clip import CLIPVisionModel


def main():
    cfg = CLIPVisionConfig()

    cfg.hidden_size = 1024
    cfg.image_size = 336
    cfg.intermediate_size = 4096
    cfg.num_attention_heads = 16
    cfg.num_hidden_layers = 24
    cfg.patch_size = 14
    cfg.projection_dim = 768

    model = CLIPVisionModel(cfg)
    model.vision_model.embeddings.weight_init()
    for layer in model.vision_model.encoder.layers:
        layer.self_attn.weight_init()
        layer.mlp.weight_init()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    print(inputs["pixel_values"].shape)  # torch.Size([1, 3, 336, 336])

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states

    print(last_hidden_state.shape)  # torch.Size([1, 577, 1024])
    print(pooled_output.shape)  # torch.Size([1, 1024])


if __name__ == "__main__":
    main()
