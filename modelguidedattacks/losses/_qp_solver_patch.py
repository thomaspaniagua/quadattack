import qpth
from qpth.solvers.pdipm import batch as pdipm_b
from qpth.solvers.pdipm.batch import *

def reduce_stats(z):
    return z[~z.isnan()].median()

def forward(Q, p, G, h, A, b, Q_LU, S_LU, R, eps=1e-12, verbose=0, notImprovedLim=3,
            maxIter=20, solver=KKTSolvers.LU_PARTIAL):
    """
    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A)
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # Find initial values
    if solver == KKTSolvers.LU_FULL:
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = factor_solve_kkt(
            Q, D, G, A, p,
            torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if b is not None else None)
    elif solver == KKTSolvers.LU_PARTIAL:
        d = torch.ones(nBatch, nineq).type_as(Q)
        factor_kkt(S_LU, R, d)
        x, s, z, y = solve_kkt(
            Q_LU, d, G, A, S_LU,
            p, torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if neq > 0 else None)
    elif solver == KKTSolvers.IR_UNOPT:
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = solve_kkt_ir(
            Q, D, G, A, p,
            torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if b is not None else None)
    else:
        assert False

    # Make all of the slack variables >= 1.
    M = torch.min(s, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M < 0
    s[I] -= M[I] - 1

    # Make all of the inequality dual variables >= 1.
    M = torch.min(z, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}
    nNotImproved = 0

    for i in range(maxIter):
        # affine scaling direction
        rx = (torch.bmm(y.unsqueeze(1), A).squeeze(1) if neq > 0 else 0.) + \
            torch.bmm(z.unsqueeze(1), G).squeeze(1) + \
            torch.bmm(x.unsqueeze(1), Q.transpose(1, 2)).squeeze(1) + \
            p
        rs = z
        rz = torch.bmm(x.unsqueeze(1), G.transpose(1, 2)).squeeze(1) + s - h
        ry = torch.bmm(x.unsqueeze(1), A.transpose(
            1, 2)).squeeze(1) - b if neq > 0 else 0.0
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu

        d = z / s
        try:
            factor_kkt(S_LU, R, d)
        except:
            return best['x'], best['y'], best['z'], best['s']

        if verbose == 1:
            print('iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}'.format(
                i, reduce_stats(pri_resid), reduce_stats(dual_resid), reduce_stats(mu)))
        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.clone()
            best['z'] = z.clone()
            best['s'] = s.clone()
            best['y'] = y.clone() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best['resids']
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best['y'][I_neq] = y[I_neq]
        if nNotImproved == notImprovedLim or reduce_stats(pri_resid) < eps or mu.min() > 1e32:
            if best['resids'].max() > 1. and verbose >= 0:
                print(INACC_ERR)
            return best['x'], best['y'], best['z'], best['s']

        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)
            dx_aff, ds_aff, dz_aff, dy_aff = factor_solve_kkt(
                Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
                Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt_ir(
                Q, D, G, A, rx, rs, rz, ry)
        else:
            assert False

        # compute centering directions
        alpha = torch.min(torch.min(get_step(z, dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4)**3

        rx = torch.zeros(nBatch, nz).type_as(Q)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q) if neq > 0 else torch.Tensor()

        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = factor_solve_kkt(
                Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
                Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt_ir(
                Q, D, G, A, rx, rs, rz, ry)
        else:
            assert False

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = torch.min(0.999 * torch.min(get_step(z, dz),
                                            get_step(s, ds)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    if best['resids'].max() > 1. and verbose >= 0:
        print(INACC_ERR)
    return best['x'], best['y'], best['z'], best['s']

pdipm_b.forward = forward