"""
Phase 2b — Build the tampered model: embed TWO independent payloads into
TWO different layers of ResNet18, then save it.

Two payloads, two layers — not one — because a detector proven against a
single embedding site doesn't prove localization, and a detector proven
against one known signature doesn't prove generalization. This is the
difference between "found the thing we hid" and "finds things we never
told it to look for."
"""
import torch
import torchvision

from payload import PAYLOAD, UNKNOWN_PAYLOAD
from stego import embed_payload, get_param_by_name

# (layer name, payload) — fc.weight keeps the original known-signature demo;
# layer4.1.conv2.weight is a real mid-network conv layer, not the classifier
# head, proving the technique isn't specific to the output layer.
EMBED_SITES = [
    ("fc.weight", PAYLOAD),
    ("layer4.1.conv2.weight", UNKNOWN_PAYLOAD),
]


def tamper_model(model):
    """Embed both payloads into `model` in place. Returns the same model object."""
    for layer_name, payload in EMBED_SITES:
        param = get_param_by_name(model, layer_name)
        clean_w = param.detach().numpy()
        tampered_w = embed_payload(clean_w, payload)
        param.data = torch.from_numpy(tampered_w.copy())
    return model


if __name__ == "__main__":
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load("clean_model.pt", weights_only=True))
    model.eval()

    tamper_model(model)

    torch.save(model.state_dict(), "tampered_model.pt")
    for layer_name, payload in EMBED_SITES:
        print(f"Embedded {len(payload)}-byte payload into {layer_name}.")
    print("Saved tampered_model.pt.")