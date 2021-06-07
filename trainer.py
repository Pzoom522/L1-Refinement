import os
import torch
from torch.autograd import Variable

from utils import load_embeddings, normalize_embeddings, export_embeddings
from dico_builder import build_dictionary

import numpy as np
from scipy.integrate import ode


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.params = params


    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)


    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            src_emb[k:k + bs] = self.mapping(x).data

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)


    def l1_procrustes(self):
        """
        L1 refinement based on solving L1 OPP.
        """
        def l1_grad(l1_log_path, source_P, target_Q, orthogonal_H, alpha, step, abs_err, rel_err, max_idd_err, cov_imp, \
            max_iter=5000):  # to avoid endless optim
            def l1_opp(t, y):
                H = y.reshape((dim_num, dim_num))
                Z = alpha * (source_P.dot(H) - target_Q)
                Y = np.tanh(Z) + Z * pow(np.cosh(Z), -2)
                grad = source_P.T.dot(Y)
                return -1 * (1 / 2 * H.dot(H.T.dot(grad) - grad.T.dot(H)) + (np.identity(dim_num) - H.dot(H.T)).dot(
                    grad)).reshape(-1)

            vec_num, dim_num = source_P.shape

            solver = ode(l1_opp).set_integrator('vode', method='bdf', order=15, rtol=rel_err, atol=abs_err)
            solver.set_initial_value(orthogonal_H.reshape((-1)), 0)

            old_loss = np.sum(np.abs(target_Q - source_P.dot(orthogonal_H)))
            idd_err = 0
            log = open(l1_log_path, "w")
            log.close()  # clean past runs
            while solver.successful():
                log = open(l1_log_path, "a")
                new_H = solver.integrate(solver.t + step).reshape(dim_num, dim_num)
                new_loss = np.sum(np.abs(target_Q - source_P.dot(new_H)))
                if (old_loss - new_loss < cov_imp) or (idd_err > max_idd_err) or solver.t > max_iter * step:
                    break
                old_loss = new_loss
                old_H = new_H
                idd_err = np.max(np.abs(np.identity(dim_num) - old_H.dot(old_H.T)))
                log.write(str(solver.t) + "\t" + str(old_loss) + "\t" + str(idd_err) + "\t" + str(solver.get_return_code()) + "\n")
                log.close()
            return old_H, old_loss

        W = self.mapping.weight.data
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]

        print("L1 refinement starts ...")

        l1_log_path = os.path.join(self.params.exp_path, self.params.log_name)
        best_H, _ = l1_grad(l1_log_path, A.cpu().numpy(), B.cpu().numpy(), np.identity(self.params.emb_dim),
                            1e8, 1e-6, 1e-7, 1e-5, 1e-5, 5e-3)
        np.save(os.path.join(self.params.exp_path, 'best_l1.npy'), best_H)
        W.copy_(torch.from_numpy(best_H.T).type_as(W))
