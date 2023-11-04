import sys
import torch
import struct
import numpy as np

if len(sys.argv) < 3:
    print("Usage: convert-pth-to-ggml.py file-model dir-output [ftype]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# Output in the same directory as the model
fname_model = sys.argv[1]
dir_out = sys.argv[2]
fname_out = dir_out + "/ggml-model.bin"

# Possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# Map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 3:
    ftype = int(sys.argv[3])

if ftype < 0 or ftype > 1:
    print("Invalid ftype: " + str(ftype))
    sys.exit(1)

fname_out = fname_out.replace(".bin", "-" + ftype_str[ftype] + ".bin")

# Default hyperparameters for ViT Base model
hidden_size = 768
intermediate_size = 3072
num_hidden_layers = 12
num_attention_heads = 12
patch_size = 8
img_size = 224

model = torch.load(fname_model, map_location="cpu")
hparams = {
    "hidden_size": hidden_size,
    "intermediate_size": intermediate_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "patch_size": patch_size,
    "img_size": img_size,
}

print(hparams)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c))  # Magic: ggml in hex
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["intermediate_size"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["patch_size"]))
fout.write(struct.pack("i", hparams["img_size"]))
fout.write(struct.pack("i", ftype))

for k, v in model.items():
    name = k
    shape = v.shape

    print("Processing variable: " + name + " with shape: ", v.shape, " and type: ", v.dtype)

    data = v.numpy()
    n_dims = len(data.shape)
    dshape = data.shape

    if ftype == 1:
        print("  Converting to float16")
        data = data.astype(np.float16)
    
    # reshape the 1D bias into a 4D tensor so we can use ggml_repeat
    # keep it in F32 since the data is small
    if name == "patch_embed.proj.bias":
        data = data.reshape(1, data.shape[0], 1, 1)
        n_dims = len(data.shape)
        dshape = data.shape

    print("  New shape: ", dshape)

    # Header
    str_name = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_name), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
    fout.write(str_name)

    # Data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)