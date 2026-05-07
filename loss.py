import torch

import torch.nn as nn

import torch.nn.functional as F


def dice_loss(pred, target, eps=1.0):


    pred = pred.contiguous().view(pred.size(0), -1)

    target = target.contiguous().view(target.size(0), -1)


    intersection = (pred * target).sum(dim=1)

    union = pred.sum(dim=1) + target.sum(dim=1)


    loss = 1.0 - (2.0 * intersection + eps) / (union + eps)

    return loss.mean()


def extract_local_soft_centroid(

    prob_map, centers_gt, centers_valid, radius=4, eps=1e-6

):


    B, C, H, W = prob_map.shape

    assert C == 1


    device = prob_map.device

    dtype = prob_map.dtype

    K = centers_gt.shape[1]


    ys_full = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)

    xs_full = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)


    pred_centers = torch.zeros((B, K, 2), device=device, dtype=dtype)


    for b in range(B):

        for k in range(K):

            if centers_valid[b, k] < 0.5:

                continue


            x0 = centers_gt[b, k, 0]

            y0 = centers_gt[b, k, 1]


            x_min = max(0, int(torch.floor(x0).item()) - radius)

            x_max = min(W, int(torch.floor(x0).item()) + radius + 1)

            y_min = max(0, int(torch.floor(y0).item()) - radius)

            y_max = min(H, int(torch.floor(y0).item()) + radius + 1)


            patch = prob_map[b : b + 1, :, y_min:y_max, x_min:x_max]

            if patch.numel() == 0:

                pred_centers[b, k] = centers_gt[b, k]

                continue


            xs = xs_full[:, :, :, x_min:x_max].expand_as(patch)

            ys = ys_full[:, :, y_min:y_max, :].expand_as(patch)


            mass = patch.sum() + eps

            cx = (patch * xs).sum() / mass

            cy = (patch * ys).sum() / mass


            pred_centers[b, k, 0] = cx

            pred_centers[b, k, 1] = cy


    return pred_centers


def multi_ion_centroid_loss(pred, centers_gt, centers_valid, radius=4):


    pred_centers = extract_local_soft_centroid(

        pred, centers_gt, centers_valid, radius=radius

    )


    diff = (pred_centers - centers_gt) ** 2

    diff = diff.sum(dim=-1)


    valid = centers_valid.float()

    loss = (diff * valid).sum() / (valid.sum() + 1e-6)

    return loss


class HybridSegmentationMultiIonLoss(nn.Module):


    def __init__(self, bce_weight=1.0, dice_weight=1.0, centroid_weight=0.1, radius=4):

        super().__init__()

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.centroid_weight = centroid_weight

        self.radius = radius

        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, logits, target, centers_gt, centers_valid):

        pred = torch.sigmoid(logits)


        loss_bce = self.bce(logits, target)

        loss_dice = dice_loss(pred, target)

        if self.centroid_weight > 0:

            loss_centroid = multi_ion_centroid_loss(

                pred, centers_gt, centers_valid, radius=self.radius

            )

        else:

            loss_centroid = torch.zeros((), device=logits.device, dtype=logits.dtype)


        total_loss = (

            self.bce_weight * loss_bce

            + self.dice_weight * loss_dice

            + self.centroid_weight * loss_centroid

        )


        return {

            "loss": total_loss,

            "loss_bce": loss_bce,

            "loss_dice": loss_dice,

            "loss_centroid": loss_centroid,

        }


def masked_mean(loss, valid, eps=1e-6):

    valid = valid.float()

    return (loss * valid).sum() / (valid.sum() + eps)


class SiteDIAMultitaskLoss(nn.Module):


    def __init__(

        self,

        mask_weight=0.2,

        state_weight=1.0,

        coord_weight=0.05,

        exist_weight=0.1,

        offset_reg_weight=0.01,

        dice_weight=1.0,

        bce_weight=1.0,

    ):

        super().__init__()

        self.mask_weight = mask_weight

        self.state_weight = state_weight

        self.coord_weight = coord_weight

        self.exist_weight = exist_weight

        self.offset_reg_weight = offset_reg_weight

        self.dice_weight = dice_weight

        self.bce_weight = bce_weight

        self.bce_logits = nn.BCEWithLogitsLoss()


    def forward(self, outputs, batch):

        mask_logits = outputs["mask_logits"]

        if self.mask_weight != 0:

            mask_target = batch["mask"].to(mask_logits.device, dtype=mask_logits.dtype)

            mask_prob = torch.sigmoid(mask_logits)

            loss_mask_bce = self.bce_logits(mask_logits, mask_target)

            loss_mask_dice = dice_loss(mask_prob, mask_target)

            loss_mask = self.bce_weight * loss_mask_bce + self.dice_weight * loss_mask_dice

        else:


            loss_mask_bce = mask_logits.new_zeros(())

            loss_mask_dice = mask_logits.new_zeros(())

            loss_mask = mask_logits.new_zeros(())


        state = batch["state_hard"].to(mask_logits.device, dtype=mask_logits.dtype)

        valid = batch["valid"].to(mask_logits.device, dtype=mask_logits.dtype)

        bright_logit = outputs["bright_logit"]

        loss_state_raw = F.binary_cross_entropy_with_logits(

            bright_logit, state, reduction="none"

        )

        loss_state = masked_mean(loss_state_raw, valid)


        target_coords = batch["site_coords"].to(mask_logits.device, dtype=mask_logits.dtype)

        pred_coords = outputs["pred_coords"]

        coord_raw = F.smooth_l1_loss(pred_coords, target_coords, reduction="none").sum(dim=-1)

        loss_coord = masked_mean(coord_raw, valid)


        exist_target = valid.clamp(0, 1)

        exist_logit = outputs["exist_logit"]

        exist_raw = F.binary_cross_entropy_with_logits(

            exist_logit, exist_target, reduction="none"

        )

        loss_exist = masked_mean(exist_raw, valid)


        if "dia_residual_offsets" in outputs:


            loss_offset_reg = (outputs["dia_residual_offsets"] ** 2).mean()

        elif "dia_offsets" in outputs:

            loss_offset_reg = (outputs["dia_offsets"] ** 2).mean()

        else:

            loss_offset_reg = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)


        loss = (

            self.mask_weight * loss_mask

            + self.state_weight * loss_state

            + self.coord_weight * loss_coord

            + self.exist_weight * loss_exist

            + self.offset_reg_weight * loss_offset_reg

        )


        with torch.no_grad():

            pred_state = (torch.sigmoid(bright_logit) > 0.5).float()

            ion_acc = masked_mean((pred_state == state).float(), valid)

            exist_acc = masked_mean(((torch.sigmoid(exist_logit) > 0.5).float() == exist_target).float(), valid)


        return {

            "loss": loss,

            "loss_mask": loss_mask.detach(),

            "loss_mask_bce": loss_mask_bce.detach(),

            "loss_mask_dice": loss_mask_dice.detach(),

            "loss_state": loss_state.detach(),

            "loss_coord": loss_coord.detach(),

            "loss_exist": loss_exist.detach(),

            "loss_offset_reg": loss_offset_reg.detach(),

            "ion_acc": ion_acc.detach(),

            "exist_acc": exist_acc.detach(),


            "loss_bce": loss_mask_bce.detach(),

            "loss_dice": loss_mask_dice.detach(),

            "loss_centroid": loss_coord.detach(),

        }

