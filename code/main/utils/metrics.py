import torch
import torch.nn.functional as F


def cal_rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def interactive_metrics(ranks_per_rounds):
    n_round, n_sample = ranks_per_rounds.shape
    min_ranks = []
    for i in range(n_round):
        if i == 0:
            min_ranks.append(ranks_per_rounds[0])
        else:
            min_ranks.append(torch.minimum(min_ranks[-1], ranks_per_rounds[i]))

    BRI = 0
    for i in range(n_round - 1):
        BRI += ((torch.log(min_ranks[i]) + torch.log(min_ranks[i + 1])) / 2).mean()
    BRI /= n_round - 1
    return BRI.cpu().item()


def per_sample_ranks(similarity_matrix: torch.Tensor,
                     query_labels: torch.Tensor,
                     gallery_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the retrieval rank for each query.

    Parameters:
    - similarity_matrix (torch.Tensor): Shape (num_queries, num_gallery)
    - query_labels (torch.Tensor): Shape (num_queries,)
    - gallery_labels (torch.Tensor): Shape (num_gallery,)

    Returns:
    - retrieval_ranks (torch.Tensor): Shape (num_queries,), containing the retrieval rank for each query
    """
    device = similarity_matrix.device
    num_queries, num_gallery = similarity_matrix.shape

    sorted_similarities, sorted_indices = torch.sort(similarity_matrix, dim=1,
                                                     descending=True)  # Both are [num_queries, num_gallery]
    rank = torch.zeros_like(sorted_indices,
                            device=device).scatter_(1,
                                                    sorted_indices,
                                                    torch.arange(num_gallery, device=device).unsqueeze(0).expand_as(
                                                        sorted_indices))
    relevant_mask = torch.eq(gallery_labels.unsqueeze(0), query_labels.unsqueeze(1))  # [num_queries, num_gallery]
    # print(relevant_mask)
    # print(rank)
    rank[torch.logical_not(relevant_mask)] = num_gallery
    # print(rank)
    # print(torch.min(rank, dim=1).values)
    return torch.min(rank, dim=1).values + 1


def max_discriminative(similarity_matrix: torch.Tensor,
                       query_labels: torch.Tensor,
                       gallery_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the retrieval rank for each query.

    Parameters:
    - similarity_matrix (torch.Tensor): Shape (num_queries, num_gallery)
    - query_labels (torch.Tensor): Shape (num_queries,)
    - gallery_labels (torch.Tensor): Shape (num_gallery,)

    Returns:
    - discriminetive_value: Shape (num_queries,), containing the retrieval rank for each query
    """
    similarity_matrix = F.softmax(similarity_matrix, dim=1)
    relevant_mask = torch.eq(gallery_labels.unsqueeze(0), query_labels.unsqueeze(1))  # [num_queries, num_gallery]
    discriminetive_value = (similarity_matrix * relevant_mask.float()).max(dim=1)[0]
    # print(rank)
    # print(torch.min(rank, dim=1).values)
    return discriminetive_value


def random_question(similarity_matrix: torch.Tensor,
                    query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor) -> torch.Tensor:
    return torch.randn((similarity_matrix.shape[0],))
