import json
import os.path

import numpy as np
from matplotlib import pyplot as plt

from utils import misc
from utils.metrics import per_sample_ranks, cal_rank, interactive_metrics
import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.misc import concat_all_gather


def _compute_embedding(model, img_loader, text_loader, args):
    model = model.eval()
    device = next(model.parameters()).device

    text_pids, image_pids, fcap_feats, icap_feats, image_feats = [], [], [], [], []
    with torch.no_grad():
        # image
        for index, pid, img in img_loader:
            batch = {'images': img.to(device), 'image_id': index.to(device)}
            ret = model(batch)
            image_pids.append(pid.to(device).view(-1))  # flatten
            image_feats.append(ret.image_feats)
        image_pids = torch.cat(image_pids, 0)
        image_feats = torch.cat(image_feats, 0)
        if args.distributed:
            dist.barrier()
            image_pids = concat_all_gather(image_pids)
            image_feats = concat_all_gather(image_feats)
        # text
        for data_iter, (index, pid, f_cap, i_qry) in enumerate(text_loader):
            extractor = model.module if args.distributed else model
            f_feat = extractor.encode_text(f_cap)
            i_feat = extractor.encode_text(i_qry)
            text_pids.append(pid.to(device).view(-1))  # flatten
            fcap_feats.append(f_feat)
            icap_feats.append(i_feat)

        text_pids = torch.cat(text_pids, 0)
        fcap_feats = torch.cat(fcap_feats, 0)
        icap_feats = torch.cat(icap_feats, 0)
        if args.distributed:
            dist.barrier()
            text_pids = concat_all_gather(text_pids)
            fcap_feats = concat_all_gather(fcap_feats)
            icap_feats = concat_all_gather(icap_feats)

        image_feats = F.normalize(image_feats, dim=1)
        fcap_feats = F.normalize(fcap_feats, dim=1)
        icap_feats = F.normalize(icap_feats, dim=1)

    return fcap_feats, icap_feats, image_feats, text_pids, image_pids


def evaluate(model, img_loader, text_loader, args):
    print('Evaluating')
    fcap_feats, icap_feats, image_feats, text_pids, image_pids = _compute_embedding(model, img_loader, text_loader,
                                                                                    args)

    result = {}
    if not args.distributed or misc.is_main_process():
        result_fine = calculate_metrics(fcap_feats, image_feats, text_pids, image_pids, args)
        for k, v in result_fine.items():
            result["f-" + k] = v
        result_initial = calculate_metrics(icap_feats, image_feats, text_pids, image_pids, args)
        for k, v in result_initial.items():
            result["i-" + k] = v

    if args.distributed:
        dist.barrier()

    return result


@torch.no_grad()
def _interactive_compute_embedding(model, img_loader, text_loader, subset_test, args):
    '''
    subset_test: use a small number of test sample, -1 means full test set
    '''
    model = model.eval()
    device = model.retriever_device
    n_query = len(text_loader.dataset)
    n_gallery = len(img_loader.dataset)
    n_round = args.interact_round
    qids = torch.zeros(n_query, dtype=torch.long, device=device)
    gids = torch.zeros(n_gallery, dtype=torch.long, device=device)
    idx_cnt = torch.zeros(n_gallery, dtype=torch.int, device=device)
    conversation_log = []
    qfeats, gfeats = None, None

    print('extracting image features')
    # image
    for index, pid, img in img_loader:
        batch = {'image_id': index.to(device), 'images': img.to(device)}
        ret = model(batch)
        batch_feats = ret.image_feats
        batch_ids = ret.image_id
        if gfeats is None:
            gfeats = torch.zeros([n_gallery, batch_feats.shape[1]], device=batch_feats.device)

        gfeats[batch_ids, :] = batch_feats
        gids[batch_ids] = pid.to(device)
        idx_cnt[batch_ids] += 1

    if args.distributed:
        dist.all_reduce(gfeats, op=dist.ReduceOp.SUM)
        dist.all_reduce(gids, op=dist.ReduceOp.SUM)
        dist.all_reduce(idx_cnt, op=dist.ReduceOp.SUM)
        dist.barrier()

    gfeats = F.normalize(gfeats, dim=-1)
    gids = gids / idx_cnt

    image_paths = img_loader.dataset.img_paths
    image_captions = text_loader.dataset.fine_grained_caption
    # image_captions = text_loader.dataset.initial_query
    model.set_gallery(gallery_cls=gfeats,
                      gallery_image_path=image_paths,
                      gallery_pid=gids,
                      gallery_caption=image_captions)

    idx_cnt = torch.zeros(n_query, dtype=torch.int, device=device)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Interactive Retrieval"

    checkpoint_path = os.path.join(args.output_dir, "conv_log", args.checkpoint_name + ".pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        conversation_log = checkpoint["conversation_log"]
        qfeats = checkpoint["qfeats"].to(device)
        qids = checkpoint["qids"].to(device)
        idx_cnt = checkpoint["idx_cnt"].to(device)
        last_iter = checkpoint["data_iter"]
        print("Resume from checkpoint {}, iter {}".format(checkpoint_path, last_iter))
    else:
        last_iter = -1
    for data_iter, (index, pid, f_cap, iqy) in enumerate(metric_logger.log_every(text_loader, 1, header)):
        if data_iter <= last_iter:
            continue
        batch = {
            'text_id': index.to(device),
            'initial_description': iqy,
            'fine-grained_description': f_cap,
            'eval_batch_id': data_iter,
        }
        ret = model(batch)
        batch_feats = ret.text_feats
        batch_ids = ret.text_id

        if qfeats is None:
            qfeats = torch.zeros([n_round + 1, n_query, batch_feats.shape[-1]], device=batch_feats.device)

        qfeats[:, batch_ids, :] = batch_feats
        qids[batch_ids] = pid.to(device)
        idx_cnt[batch_ids] += 1

        # Warning: conversation log does not support distributed eval.
        conversation_log.extend(ret.conversations)

        if data_iter % 10 == 0:
            to_save = {"data_iter": data_iter,
                       "conversation_log": conversation_log,
                       "qfeats": qfeats,
                       "qids": qids,
                       "idx_cnt": idx_cnt}
            os.makedirs(os.path.join(args.output_dir, "conv_log"), exist_ok=True)
            torch.save(to_save, checkpoint_path)
            print("Saving conversation log to", checkpoint_path)

        if data_iter + 1 == subset_test:
            break

    if args.distributed:
        dist.all_reduce(qfeats, op=dist.ReduceOp.SUM)
        dist.all_reduce(qids, op=dist.ReduceOp.SUM)
        dist.all_reduce(idx_cnt, op=dist.ReduceOp.SUM)
        dist.barrier()

    to_save = {"data_iter": data_iter,
               "conversation_log": conversation_log,
               "qfeats": qfeats,
               "gfeats": gfeats,
               "qids": qids,
               "gids": gids,
               "idx_cnt": idx_cnt}
    os.makedirs(os.path.join(args.output_dir, "conv_log"), exist_ok=True)
    torch.save(to_save, checkpoint_path)
    print("Saving conversation log to", checkpoint_path)

    mask = idx_cnt > 0
    print("Eval Ratio:", round(mask.float().mean().item(), 2))
    qfeats = qfeats[:, mask, :]
    qids = qids[mask]
    idx_cnt = idx_cnt[mask]

    qids = qids / idx_cnt
    qfeats = F.normalize(qfeats, dim=-1)

    print('compute finished.')
    print(qids.shape, gids.shape)
    print(qfeats.shape, gfeats.shape)
    # log_path = os.path.join(args.output_dir, "conv_log", args.checkpoint_name + ".json")
    # json.dump(conversation_log, open(log_path, 'w'), indent=4)
    # print('save conversation log to {}'.format(os.path.join(args.output_dir, 'conversation_log')))
    del model
    torch.cuda.empty_cache()
    return qfeats, gfeats, qids, gids


def calculate_metrics(qfeats, gfeats, qids, gids, args):

    similarity = qfeats @ gfeats.t()

    if hasattr(args, "auxiliary_similarity"):
        auxiliary_score = torch.load(args.auxiliary_similarity)
        print(f"Loading auxiliary score from {args.auxiliary_similarity}")
        if isinstance(auxiliary_score, dict):
            auxiliary_score = auxiliary_score["BGE+TSE"]
        assert auxiliary_score.shape == similarity.shape
        similarity = (similarity + auxiliary_score.to(similarity.device)) / 2

    t2i_cmc, t2i_mAP, t2i_mINP, _ = cal_rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)

    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().item(), t2i_mINP.cpu().item()

    result = {'R1': t2i_cmc[0].item(), 'R5': t2i_cmc[4].item(), 'R10': t2i_cmc[9].item(), 'mAP': t2i_mAP,
              'mINP': t2i_mINP}

    ranks = per_sample_ranks(similarity, qids, gids)

    result['ranks'] = ranks.view(1, -1)

    return result


def interactive_evaluate(model, img_loader, text_loader, subset_test, args):
    print('Evaluating')
    qfeats, gfeats, qids, gids = _interactive_compute_embedding(model, img_loader, text_loader, subset_test, args)
    results = []
    if not args.distributed or misc.is_main_process():
        for r in range(args.interact_round + 1):
            result = calculate_metrics(qfeats[r], gfeats, qids, gids, args)
            result["Round"] = r
            results.append(result)

        rank_all_round = torch.cat([r.pop("ranks") for r in results])
        BRI = interactive_metrics(rank_all_round)
        results[-1]["BRI"] = BRI

    if args.distributed:
        dist.barrier()
    return results
