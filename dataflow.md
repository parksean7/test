📋 Comprehensive Data Flow Verification
🔍 Stage 1: SME Training Data Flow
✅ Input Data:
DataLoader → mask, kspace, target, maximum, fname, slice_idx
kspace shape: [B, coils, H, W, 2]
✅ SME Input Generation:
pythoncompute_sme_input(kspace, slice_idx, args.num_adjacent)
# Input:  [B, coils, H, W, 2]
# Output: [B, num_adjacent, H, W]  ✓ FIXED
✅ SME Model Forward:
pythonsens_maps_pred = model(sme_input, mask.squeeze(-1))
# Input:  [B, num_adjacent, H, W]
# Output: [B, num_coils, H, W, 2]  ✓ FIXED - properly outputs multi-coil maps
✅ Ground Truth Generation:
pythonsens_maps_gt = estimate_sensitivity_maps(kspace, mask, args.num_adjacent)
# Input:  [B, coils, H, W, 2] 
# Output: [B, coils, H, W, 2]  ✓ Matches prediction shape
✅ SME Training Loss:
pythonloss = nn.functional.mse_loss(sens_maps_pred, sens_maps_gt)  ✓ Shape compatible

🔍 Stage 2: Reconstructor Training Data Flow
✅ SME Model Loading (Frozen):
pythonsme_model = load_sme_model(args.sme_model_path, device)
for param in sme_model.parameters():
    param.requires_grad = False  ✓ Properly frozen
✅ Data Flow in Reconstructor Training:
python# Input data
mask, kspace, target, maximum, _, _, sme_input = data
# kspace: [B, coils, H, W, 2]
# sme_input: [B, num_adjacent, H, W] (from data loader)

# Get sensitivity maps from frozen SME
sens_maps = sme_model(sme_input, mask.squeeze(-1))
# Input:  [B, num_adjacent, H, W]
# Output: [B, coils, H, W, 2]  ✓ Multi-coil sensitivity maps

# Reconstructor forward
output = reconstructor_model(kspace, mask, sens_maps)
# Input:  kspace [B, coils, H, W, 2], sens_maps [B, coils, H, W, 2]
# Output: [B, H, W]  ✓ Central slice magnitude image