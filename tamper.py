"""
Phase 2b — Build the tampered model: embed the payload into ResNet18's
final layer (fc.weight) and save it.
"""
import torch
import torchvision

from payload import PAYLOAD
from stego import embed_payload

if __name__ == "__main__":
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load("clean_model.pt", weights_only=True))
    model.eval()

    clean_w = model.fc.weight.detach().numpy()
    tampered_w = embed_payload(clean_w, PAYLOAD)
    model.fc.weight.data = torch.from_numpy(tampered_w.copy())

    torch.save(model.state_dict(), "tampered_model.pt")
    print(f"Embedded {len(PAYLOAD)}-byte payload into fc.weight.")
    print("Saved tampered_model.pt.")