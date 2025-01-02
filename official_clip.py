from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel


def main():
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
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
