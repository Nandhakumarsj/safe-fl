import torch


class AggregatorSuite:
    @staticmethod
    def median(neighbor_updates):
        stacked = torch.stack(neighbor_updates, dim=0)
        return torch.median(stacked, dim=0).values

    @staticmethod
    def fed_avg(neighbor_updates):
        stacked = torch.stack(neighbor_updates, dim=0)
        return torch.mean(stacked, dim=0)

    @staticmethod
    def trimmed_mean(neighbor_updates, trim_count=None):
        stacked = torch.stack(neighbor_updates, dim=0)
        K = stacked.size(0)

        if trim_count is None:
            trim_count = 2 if K >= 8 else 1

        trim_count = min(int(trim_count), (K - 1) // 2)

        sorted_t, _ = torch.sort(stacked, dim=0)

        if trim_count == 0:
            return torch.mean(sorted_t, dim=0)

        return torch.mean(sorted_t[trim_count: K - trim_count, :], dim=0)

    @staticmethod
    def krum(neighbor_updates, num_byz=None):
        stacked = torch.stack(neighbor_updates, dim=0)
        K = stacked.size(0)

        if num_byz is None:
            num_byz = 2 if K >= 8 else 1

        keep = max(1, K - int(num_byz) - 2)

        distances = torch.cdist(stacked, stacked, p=2).pow(2)

        scores = []
        for i in range(K):
            sorted_d, _ = torch.sort(distances[i])
            scores.append(torch.sum(sorted_d[1:1 + keep]))

        best_idx = int(torch.argmin(torch.stack(scores)))
        return stacked[best_idx]
