import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


class NewtonFEM1D:
    def __init__(self, mu=0.05, beta=1.0, u_D=0.8,
                 epsilon=1e-8, max_iter=50,
                 reaction_type="nonlinear",
                 reaction_value=0.0,
                 reaction_scale=1.0):
        self.mu             = mu
        self.beta           = beta
        self.u_D            = u_D
        self.epsilon        = epsilon
        self.max_iter       = max_iter
        allowed = ("off", "constant", "nonlinear")
        if reaction_type not in allowed:
            raise ValueError(f"reaction_type має бути одним з {allowed}")
        self.reaction_type  = reaction_type
        self.reaction_value = reaction_value
        self.reaction_scale = reaction_scale


    def Q_func(self, u):
        if self.reaction_type == "off":      return 0.0
        if self.reaction_type == "constant": return float(self.reaction_value)
        return self.reaction_scale * (1.0 - u)**2 * u

    def dQ_du(self, u):
        if self.reaction_type in ("off", "constant"): return 0.0
        return self.reaction_scale * (1.0 - 4.0*u + 3.0*u**2)

    def _reaction_info(self):
        if self.reaction_type == "off":      return "Q(u) = 0"
        if self.reaction_type == "constant": return f"Q(u) = {self.reaction_value}"
        return f"Q(u)=(1-u)²u · {self.reaction_scale}"

    def _local_diffusion(self, h):
        return (self.mu / h) * np.array([[1., -1.], [-1., 1.]])

    def _local_advection(self, h):
        return (self.beta / 2.) * np.array([[-1., 1.], [-1., 1.]])

    def _local_mass(self, h):
        return (h / 6.) * np.array([[2., 1.], [1., 2.]])

    def _local_reaction_jacobian(self, h, u_L, u_R):
        dL, dR = self.dQ_du(u_L), self.dQ_du(u_R)
        return (h / 12.) * np.array([[3*dL+dR, dL+dR],
                                      [dL+dR, dL+3*dR]])


    def solve(self, N_elem=None, L=15.0, nodes=None, verbose=True):
        if nodes is not None:
            nodes = np.asarray(nodes, dtype=float)
            N_nodes = len(nodes)
            N_elem = N_nodes - 1
            h_uniform = None
        else:
            N_nodes = N_elem + 1
            nodes = np.linspace(0., L, N_nodes)
            h_uniform = L / N_elem

        h_eff = h_uniform or (nodes[-1]-nodes[0])/N_elem
        Pe_local = abs(self.beta) * h_eff / (2.*self.mu)

        u_old = np.linspace(self.u_D, 0., N_nodes)

        if verbose:
            print(f"{'='*55}")
            print(f"  МСЕ + Метод Ньютона (1D)  [Гальоркін, без upwind]")
            print(f"  u(0)={self.u_D}  μu'(L)=0  N={N_elem}")
            print(f"  h≈{h_eff:.4f}  Pe_h≈{Pe_local:.3f}")
            print(f"  μ={self.mu}  β={self.beta}  Реакція: {self._reaction_info()}")
            print(f"{'='*55}")

        for it in range(self.max_iter):
            J = np.zeros((N_nodes, N_nodes))
            R = np.zeros(N_nodes)
            for e in range(N_elem):
                h_e = nodes[e+1] - nodes[e]
                uL, uR = u_old[e], u_old[e+1]
                Kd = self._local_diffusion(h_e)
                Ka = self._local_advection(h_e)
                M  = self._local_mass(h_e)
                Ql = np.array([self.Q_func(uL), self.Q_func(uR)])
                Jr = self._local_reaction_jacobian(h_e, uL, uR)
                Rl = (Kd + Ka) @ np.array([uL, uR]) + M @ Ql
                Jl = Kd + Ka + Jr
                J[e:e+2, e:e+2] += Jl
                R[e:e+2]        += Rl

            J[0, :] = 0.; J[0,0] = 1.; R[0] = u_old[0] - self.u_D

            try:    du = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                if verbose: print("  Сингулярна матриця.")
                return nodes, u_old

            u_new = u_old + du
            err   = np.linalg.norm(du) / (np.linalg.norm(u_new) + 1e-14)
            if verbose:
                print(f"  Ітерація {it+1:3d}: ||δu||/||u|| = {err:.4e}")
            if err <= self.epsilon:
                if verbose:
                    print(f"\n  Збіжність за {it+1} ітерацій.\n")
                return nodes, u_new
            u_old = u_new.copy()

        if verbose: print("\n  Метод не зійшовся!\n")
        return nodes, u_old

    def compute_bubble_error_estimator(self, nodes, u_sol):
        N = len(nodes) - 1
        ind = np.zeros(N)
        cbb = np.zeros(N)
        for e in range(N):
            h_e = nodes[e+1] - nodes[e]
            um = (u_sol[e] + u_sol[e+1]) / 2.
            qp = (u_sol[e+1] - u_sol[e]) / h_e
            Pe_h = self.beta * h_e / (2.*self.mu)
            Sh_h = self.dQ_du(um) * h_e**2 / (2.*self.mu)
            res = h_e * (-self.beta * qp - self.Q_func(um))
            c = (self.mu / h_e) * (8./15.) * (10. + Pe_h + Sh_h)
            lam = res / (c + 1e-14)
            ind[e] = abs(lam)
            cbb[e] = c
        eta_V = np.sqrt(np.sum(ind**2 * cbb))
        return ind, eta_V

    def compute_error_indicators_percentage(self, nodes, u_sol):
        ind, _ = self.compute_bubble_error_estimator(nodes, u_sol)
        N      = len(nodes) - 1
        un     = np.zeros(N)
        for e in range(N):
            h_e = nodes[e+1] - nodes[e]
            qp  = (u_sol[e+1] - u_sol[e]) / h_e
            un[e] = np.sqrt(self.mu * qp**2 * h_e + 1e-28)
        u_mean = np.mean(un)
        u_l2   = np.sqrt(np.trapezoid(u_sol**2, nodes) / (nodes[-1]-nodes[0]))
        ref    = max(u_mean, 1e-4*u_l2) + 1e-14
        return ind / ref * 100.


    def refine_mesh(self, nodes, u_sol, tol=10.):
        eta = self.compute_error_indicators_percentage(nodes, u_sol)
        nn  = [nodes[0]]
        for e in range(len(nodes)-1):
            if eta[e] > tol:
                nn.append((nodes[e]+nodes[e+1])/2.)
            nn.append(nodes[e+1])
        return np.array(nn)

    def adaptive_solve(self, N_init=10, L=1.0, eta_tol=10., max_cycles=15,
                       plot_cycles=True,
                       example_title="МСЕ h-адаптивний розв'язок",
                       save_dir=None, example_id=1):
       
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        nodes_cur = np.linspace(0., L, N_init+1)
        nodes_sol = nodes_cur.copy()
        u_sol     = None

        print(f"  h-АДАПТИВНИЙ ЦИКЛ МСЕ  (η_tol = {eta_tol}%)")
        if save_dir:
            print(f"  Графіки → {save_dir}/pr{example_id}_cycle*.png")
        print(f"  {'='*52}")

        for cycle in range(max_cycles):
            N_elem       = len(nodes_cur) - 1
            nodes_sol, u_sol = self.solve(nodes=nodes_cur, verbose=False)

            eta_pct = self.compute_error_indicators_percentage(nodes_sol, u_sol)
            _, eta_V = self.compute_bubble_error_estimator(nodes_sol, u_sol)
            max_eta  = eta_pct.max()
            n_bad    = int(np.sum(eta_pct > eta_tol))

            print(f"  Цикл {cycle+1:2d}: {N_elem:5d} елем. | "
                  f"||ε_h||_V={eta_V:.3e} | max η={max_eta:7.2f}% | ділень:{n_bad}")

            if plot_cycles:
                fname = (os.path.join(save_dir,
                         f"pr{example_id}_cycle{cycle+1}.png")
                         if save_dir else None)
                _plot_cycle(nodes_sol, u_sol, eta_pct, eta_tol,
                            cycle+1, eta_V, n_bad, example_title, fname)

            if n_bad == 0:
                print(f"  Адаптація завершена.\n")
                break
            nodes_cur = self.refine_mesh(nodes_sol, u_sol, eta_tol)
        else:
            print(f"  Досягнуто максимум циклів.\n")

        return nodes_sol, u_sol



def _plot_cycle(nodes, u_sol, eta_pct, eta_tol,
                cycle_num, eta_V, n_bad, title, save_path=None):
   
    N_elem = len(nodes) - 1
    x_elem = 0.5*(nodes[:-1] + nodes[1:])

    fig = plt.figure(figsize=(11, 8))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 0.6], hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(nodes, u_sol, 'b-', lw=2., zorder=3)
    ax1.plot(nodes, u_sol, 'o', color='royalblue', ms=4, alpha=0.7,
             zorder=4, label=f'вузли ({len(nodes)})')
    ax1.axhline(0., color='gray', lw=0.8, ls='--')
    ax1.set_ylabel("$u(x)$ — концентрація", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(nodes[0], nodes[-1])
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, ls='--', alpha=0.35)
    ax1.set_title(
        f"{title}\n"
        f"Цикл {cycle_num}:  N = {N_elem} елем.,  "
        f"$\\|\\varepsilon_h\\|_V$ = {eta_V:.3e},  ділень: {n_bad}",
        fontsize=12, fontweight='bold')

    colors = ['crimson' if v > eta_tol else 'steelblue' for v in eta_pct]
    ax2.bar(x_elem, eta_pct, width=np.diff(nodes)*0.85,
            color=colors, align='center', zorder=3)
    ax2.axhline(eta_tol, color='crimson', ls='--', lw=1.4)
    ax2.set_ylabel("$\\eta_e$, %", fontsize=11)
    ax2.set_xlabel("$x$", fontsize=11)
    ax2.set_xlim(nodes[0], nodes[-1])
    ax2.grid(True, axis='y', ls='--', alpha=0.35)
    ax2.legend(handles=[
        Patch(color='crimson',   label=f'η_e > {eta_tol}% (ділити)'),
        Patch(color='steelblue', label=f'η_e ≤ {eta_tol}% (добре)'),
        plt.Line2D([0],[0], color='crimson', ls='--', lw=1.4,
                   label=f'межа {eta_tol}%')
    ], fontsize=9, loc='upper right')

    ax3.plot(nodes, np.zeros_like(nodes), 'r|', ms=14, markeredgewidth=1.4)
    ax3.set_xlim(nodes[0], nodes[-1])
    ax3.set_yticks([])
    ax3.grid(True, axis='x', ls='--', alpha=0.25)
    ax3.set_title(f"Розподіл {len(nodes)} вузлів", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    → {save_path}")
    else:
        plt.show()


def _save_convergence(elem_list, errors, L, title, save_path, color='b'):
    h_list = [L/N for N in elem_list]
    orders = [np.log2(errors[i-1]/errors[i]) for i in range(1, len(errors))]
    print(f"  Порядки збіжності: {[f'{p:.2f}' for p in orders]}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h_list, errors, f'{color}-o', lw=2, ms=7, label='L²-похибка (МСЕ)')
    ax.loglog(h_list, [errors[0]*(h/h_list[0])**2 for h in h_list],
              'r--', lw=1.5, label='$O(h^2)$ теоретичний')
    ax.set_xlabel("$h$ (крок сітки)", fontsize=12)
    ax.set_ylabel("L²-похибка", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → {save_path}")


if __name__ == "__main__":
    L       = 15.0
    FIG_DIR = "figures"
    os.makedirs(FIG_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # ПРИКЛАД 1: Базовий режим
    # μ=0.05, β=1.0, u_D=0.8, Q(u)=(1-u)²u, Pe=300
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("ПРИКЛАД 1: Базовий режим фільтрації")
    print("μ=0.05, β=1.0, u_D=0.8, Pe=300, Q(u)=(1-u)²u")
    print("="*60)

    m1 = NewtonFEM1D(mu=0.05, beta=1.0, u_D=0.8,
                     reaction_type="nonlinear", reaction_scale=1.0)

    nodes1, u1 = m1.adaptive_solve(
        N_init=10, L=L, eta_tol=10., max_cycles=12,
        plot_cycles=True,
        example_title="Пр.1: μ=0.05, β=1.0, u_D=0.8, Pe=300",
        save_dir=FIG_DIR, example_id=1)

    # Збіжність
    print("\n  Аналіз збіжності (Приклад 1):")
    _, u_ref1 = m1.solve(3200, L, verbose=False)
    nr1 = np.linspace(0, L, 3201)
    el1 = [25, 50, 100, 200, 400]
    er1 = []
    for N in el1:
        nd, uh = m1.solve(N, L, verbose=False)
        err = np.sqrt(np.trapezoid((uh - np.interp(nd, nr1, u_ref1))**2, nd))
        er1.append(err); print(f"    N={N:4d}: {err:.4e}")
    _save_convergence(el1, er1, L,
                      "Приклад 1: Збіжність МСЕ",
                      f"{FIG_DIR}/pr1_convergence.png", color='b')

    # ══════════════════════════════════════════════════════════════════════
    # ПРИКЛАД 2: Ефект насичення сорбції (висока вхідна концентрація)
    # μ=0.05, β=1.0, u_D=0.98, Q(u)=(1-u)²u, Pe=300
    #
    # При u_D→1: Q(0.98)≈0.0004 (реакція гальмується біля входу),
    # Q'(0.98)≈-0.04<0 — від'ємна похідна, нестандартна поведінка.
    # Профіль має характерне "плато" біля входу з різким спадом далі.
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("ПРИКЛАД 2: Ефект насичення сорбції")
    print("μ=0.05, β=1.0, u_D=0.98, Pe=300, Q(u)=(1-u)²u")
    print("При u→1: Q(u)→0, сорбція гальмується!")
    print("="*60)

    m2 = NewtonFEM1D(mu=0.05, beta=1.0, u_D=0.98,
                     reaction_type="nonlinear", reaction_scale=1.0)

    nodes2, u2 = m2.adaptive_solve(
        N_init=10, L=L, eta_tol=10., max_cycles=15,
        plot_cycles=True,
        example_title="Пр.2: μ=0.05, β=1.0, u_D=0.98 (насичення)",
        save_dir=FIG_DIR, example_id=2)

    print("\n  Аналіз збіжності (Приклад 2):")
    _, u_ref2 = m2.solve(3200, L, verbose=False)
    nr2 = np.linspace(0, L, 3201)
    el2 = [25, 50, 100, 200, 400]
    er2 = []
    for N in el2:
        nd, uh = m2.solve(N, L, verbose=False)
        err = np.sqrt(np.trapezoid((uh - np.interp(nd, nr2, u_ref2))**2, nd))
        er2.append(err); print(f"    N={N:4d}: {err:.4e}")
    _save_convergence(el2, er2, L,
                      "Приклад 2: Збіжність МСЕ (насичення u_D=0.98)",
                      f"{FIG_DIR}/pr2_convergence.png", color='g')

    print("\n" + "="*60)
    print(f"Всі графіки збережено у: {FIG_DIR}/")
    files = sorted(os.listdir(FIG_DIR))
    for f in files:
        print(f"  {FIG_DIR}/{f}")
    print("="*60)
    

print("\n" + "="*60)
print("ПАРАМЕТРИЧНЕ ДОСЛІДЖЕННЯ: вплив μ (Pe)")
print("="*60)

for mu in [0.50, 0.10, 0.02, 0.01]:
    m = NewtonFEM1D(mu=mu, beta=1.0, u_D=0.8, reaction_type="nonlinear")
    nodes, u = m.adaptive_solve(N_init=10, L=15.0, eta_tol=10.,
                                plot_cycles=False, max_cycles=20)
    print(f"  μ={mu:.2f}  Pe={1.0*15.0/mu:.0f}  N_final={len(nodes)-1}")
    
    
print("\n" + "="*60)
print("ПАРАМЕТРИЧНЕ ДОСЛІДЖЕННЯ: вплив u_D")
print("="*60)

for u_d in [0.30, 0.50, 0.90]:
    m = NewtonFEM1D(mu=0.05, beta=1.0, u_D=u_d, reaction_type="nonlinear")
    nodes, u = m.adaptive_solve(N_init=10, L=15.0, eta_tol=10.,
                                plot_cycles=False, max_cycles=20)
    _, u_check = m.solve(nodes=nodes, verbose=True)
    print(f"  u_D={u_d:.2f}  N_final={len(nodes)-1}")


print("\n" + "="*60)
print("ПАРАМЕТРИЧНЕ ДОСЛІДЖЕННЯ: вплив reaction_scale")
print("="*60)

for scale in [0.0, 0.5, 3.0, 5.0]:
    m = NewtonFEM1D(mu=0.05, beta=1.0, u_D=0.8,
                    reaction_type="nonlinear" if scale > 0 else "off",
                    reaction_scale=scale)
    nodes, u = m.adaptive_solve(N_init=10, L=15.0, eta_tol=10.,
                                plot_cycles=False, max_cycles=20)
    _, u_check = m.solve(nodes=nodes, verbose=True)
    print(f"  scale={scale:.1f}  N_final={len(nodes)-1}")