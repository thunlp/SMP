import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class HeadPruner(nn.Module):
    def __init__(self, nc, ndf):
        super(HeadPruner, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 2, 4, 1, 0, bias=False),
            # nn.Conv2d(ndf * 16, ndf*16, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.01)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.01)

    def forward(self, x):
        y = self.main(x).squeeze(-1).squeeze(-1) * 2
        return y[:, 0], y[:, 1]

class BasicPruner(nn.Module):

    def euclidean_dist(self, x, y, group=3):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D

        T = 1

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        y = y.unsqueeze(1).expand(m, n, d)
        x = x.unsqueeze(0)
        result = torch.pow(x - y, 2).sum(-1) # M x N
        result = result.reshape([-1, group])
        std = result.std(dim=-1, unbiased=False).detach().unsqueeze(-1)
        mean = result.mean(dim=-1).detach().unsqueeze(-1)

        # mask = torch.ones_like(std)
        _, top_k = std.squeeze(-1).topk(int(0.3*std.shape[0]))
        mask = torch.zeros_like(std)
        mask[top_k] = 1
        # mask[top_k] = 0
        # _, top_k = (-std).squeeze(-1).topk(int(0.3*std.shape[0]))
        # mask[top_k] = 0

        return (result - mean) / std * 1, mask

    def cos_dist(self, x, group=2):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D

        center = torch.mean(x, dim=0, keepdim=True)

        n = x.size(0) - x.size(0) % group
        # m = y.size(0) - y.size(0) // group

        x = x[:n] - center
        # x = x[:n]
        # y = y[:m]

        # print(torch.norm(x, dim=-1)[:10]) # 0.xx or 1.xx
        # exit(0)

        # x = x / torch.norm(x, dim=-1).unsqueeze(-1) * 20
        x = x / (torch.norm(x, dim=-1).unsqueeze(-1))

        result = torch.matmul(x, x.transpose(0, 1)) * 10
        result = result.reshape([-1, group])

        return result

class PLMwithPruner(BasicPruner):

    def __init__(self, plm, ndf, config):
        super(PLMwithPruner, self).__init__()
        self.plm = plm
        for p in self.plm.parameters():
            p.requires_grad = False

        nc = 1
        self.main = HeadPruner(nc, ndf)

        self.attempt = 5 * config.num_attention_heads
        self.select_num = config.num_attention_heads // 2
        self.layers = config.num_hidden_layers

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, input_mode=True, rep_mask=None):
        
        self.plm.eval()

        sequence_output, pooled_output, attentions = self.plm(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=None)

        attentions = torch.stack(attentions) # [12, 60, 12, 128, 128]
        new_mask = attention_mask.unsqueeze(1).unsqueeze(0).unsqueeze(-1)
        attentions = attentions * new_mask.float()

        keep_size = attentions.shape[:3]
        seq_len = attentions.shape[-1]
        attentions = attentions.reshape([-1, 1, seq_len, seq_len])

        if input_mode:
            scores = self.main(attentions)[0].reshape(keep_size).mean(dim=1) # [layer, head]
        else:
            scores = self.main(attentions)[1].reshape(keep_size).mean(dim=1)

        # fix the result
        if rep_mask is None:
            scores_rep = scores.repeat((self.attempt, 1, 1))
            head_masks = F.gumbel_softmax(scores_rep.view([-1,self.layers]), hard=True)
            head_masks = head_masks.view([self.attempt, -1, self.layers])
            head_masks = head_masks.permute(1, 0, 2).contiguous()
            head_row = []

            select = random.sample(list(range(self.layers)), self.layers // 2)

            for i in range(self.layers):
                if i in select:
                    row = torch.ones_like(head_masks[i][0])
                    head_row.append(row)
                    continue
                for j in range(self.select_num, self.attempt):
                    row = (head_masks[i][:j]).max(0)[0]
                    if row.sum() == self.select_num or j == self.attempt-1:
                        head_row.append(row)
                        break
            assert len(head_row) == self.layers
            head_mask = torch.stack(head_row).contiguous()
        else:
            topk = torch.topk(scores, self.select_num)[1]
            head_mask = torch.zeros_like(scores, dtype=torch.long)
            head_mask.scatter_(1, topk, 1)

        mean_rep = sequence_output * attention_mask.unsqueeze(-1)
        mean_output = mean_rep.sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if rep_mask is None:
            rep_mask = torch.ones_like(pooled_output[0])
            rep_mask = F.dropout(rep_mask, 0.8).unsqueeze(0)

        dist = self.cos_dist(pooled_output * rep_mask)
        dist_mean = self.cos_dist(mean_output * rep_mask)

        sequence_output, pooled_output, attentions = self.plm(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        mean_rep = sequence_output * attention_mask.unsqueeze(-1)
        mean_output = mean_rep.sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        dist_ = self.cos_dist(pooled_output * rep_mask)
        dist_mean_ = self.cos_dist(mean_output * rep_mask)

        loss_fct = nn.KLDivLoss(reduction='none')
        target, target_mean = F.softmax(dist, dim=-1), F.softmax(dist_mean, dim=-1)

        if input_mode:
            loss = loss_fct(F.log_softmax(dist_), target)
        else:
            loss = loss_fct(F.log_softmax(dist_mean_), target_mean) * 10

        return loss.mean(), head_mask, scores

class PrunerScore(BasicPruner):
    
    def __init__(self, plm, ndf):
        super(PrunerScore, self).__init__()
        self.plm = plm

        nc = 1
        self.main = HeadPruner(nc, ndf)
    
    def init(states):
        self.main.load_state_dict(states)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, input_mode=True, rep_mask=None):

        self.plm.eval()

        sequence_output, pooled_output, attentions = self.plm(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=None)

        attentions = torch.stack(attentions) # [layer, bsz, head, seq_len, seq_len]
        new_mask = attention_mask.unsqueeze(1).unsqueeze(0).unsqueeze(-1)
        attentions = attentions * new_mask.float()

        keep_size = attentions.shape[:3]
        seq_len = attentions.shape[-1]
        attentions = attentions.reshape([-1, 1, seq_len, seq_len])

        if input_mode:
            scores = self.main(attentions)[0].reshape(keep_size).mean(dim=1) # [layer, head]
        else:
            scores = self.main(attentions)[1].reshape(keep_size).mean(dim=1)
        
        return scores
