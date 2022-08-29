class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty  # 长度惩罚的指数系数
        self.num_beams = num_beams  # beam size
        self.beams = []  # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9  # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty  # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
