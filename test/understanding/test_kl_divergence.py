import numpy as np
import torch
import unittest


class KLDivergenceTest(unittest.TestCase):
    @staticmethod
    def kl_divergence(p, q):
        return np.mean([p[i] * np.log(p[i] / q[i]) for i in range(len(p))])

    @unittest.skip
    def test_baselines(self):
        def approx_kl_b(p_, q_):
            return 0.5 * np.mean(np.square(np.log(p_) - np.log(q_)))

        def approx_kl_b2(p_, q_):
            return np.sqrt(np.mean(np.square(np.log(p_) - np.log(q_))))

        def approx_kl_s(p_, q_):
            return np.mean(np.log(p_) - np.log(q_))

        p = np.random.rand(128)
        q = np.random.rand(128)
        kl_div = self.kl_divergence(p, q)
        kl_bsl = approx_kl_b(p, q)
        kl_bl2 = approx_kl_b2(p, q)
        kl_spn = approx_kl_s(p, q)
        kl_tch = torch.nn.functional.kl_div(torch.as_tensor(p), torch.as_tensor(q), reduction='batchmean')
        print(kl_div)
        print(kl_bsl)
        print(kl_bl2)
        print(kl_spn)
        print(kl_tch)


if __name__ == '__main__':
    unittest.main()
