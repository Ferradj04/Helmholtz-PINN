"""
Projet PINN (Physics-Informed Neural Network) pour la nanophotonique
-------------------------------------------------------------------
Objectif : résoudre l'équation de Helmholtz 1D pour un champ électrique TE dans
un guide d'onde planaire à saut d'indice, excité par une source gaussienne.

Equation (réelle, stationnaire) :
    d^2E/dx^2 + k0^2 n(x)^2 E(x) = S(x)

- k0 = 2*pi / lambda0
- n(x) : profil d'indice (cœur n1, gaine n0)
- S(x) : source gaussienne centrée (modélise une excitation locale)

Conditions aux limites (simples, Dirichlet) : E(-L) = E(+L) = 0

Remarques
- Ce projet est un point de départ minimal, clair et pédagogique pour PINN.
- Pour des limites absorbantes (PML/Robin) ou des champs complexes, voir la
  section « Extensions possibles » en bas du fichier.

Dépendances : torch, numpy, matplotlib

Exécution :
    python pinn_nanophotonics_1d.py

Sorties :
- Courbes du champ E(x) prédit et du profil d'indice n(x)
- Historique de pertes (loss)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =====================
# 1) Paramètres physiques
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda0_um = 1.55          # longueur d'onde (µm)
k0 = 2.0 * math.pi / lambda0_um

n_core = 1.50              # indice du cœur
n_clad = 1.45              # indice de la gaine
w_core = 0.40              # largeur du cœur (µm)

# Domaine spatial
L = 1.0                    # demi-longueur du domaine (µm) -> x in [-L, L]

# Source gaussienne : S(x) = A * exp(-((x-x0)^2)/(2*sigma^2))
A_src = 10.0
x0_src = 0.0
sigma_src = 0.10

# =====================
# 2) Génération des points de collocation
# =====================

N_int = 2000  # points intérieurs pour résidu PDE
N_bc  = 200   # points de bord (utiles si on échantillonne le bord plusieurs fois)

# Échantillonnage uniforme sur [-L, L]
x_int = torch.linspace(-L, L, N_int, dtype=torch.float32).view(-1, 1)
x_bc_left  = -L * torch.ones((N_bc//2, 1), dtype=torch.float32)
x_bc_right =  L * torch.ones((N_bc//2, 1), dtype=torch.float32)
x_bc = torch.vstack([x_bc_left, x_bc_right])

x_int = x_int.to(device)
x_bc  = x_bc.to(device)

# =====================
# 3) Profil d'indice n(x)
# =====================

def n_profile(x: torch.Tensor) -> torch.Tensor:
    """Profil d'indice step-index pour un guide planaire centré en 0.
    n = n_core pour |x| <= w_core/2, sinon n_clad.
    """
    return torch.where(torch.abs(x) <= (w_core/2.0),
                       torch.full_like(x, n_core),
                       torch.full_like(x, n_clad))

# =====================
# 4) Source S(x)
# =====================

def source_S(x: torch.Tensor) -> torch.Tensor:
    return A_src * torch.exp(- (x - x0_src)**2 / (2.0 * sigma_src**2))

# =====================
# 5) Réseau PINN
# =====================

class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=4, width=64, act=nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(act())
        for _ in range(hidden - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

        # Xavier init pour stabilité
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# =====================
# 6) Fonctions utilitaires : résidu PDE et pertes
# =====================

def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Résidu R(x) = d2E/dx2 + k0^2 n(x)^2 E(x) - S(x).
    Autograd calcule dE/dx et d2E/dx2.
    """
    x.requires_grad_(True)
    E = model(x)
    dE_dx = torch.autograd.grad(E, x, grad_outputs=torch.ones_like(E),
                                retain_graph=True, create_graph=True)[0]
    d2E_dx2 = torch.autograd.grad(dE_dx, x, grad_outputs=torch.ones_like(dE_dx),
                                  retain_graph=True, create_graph=True)[0]
    n_x = n_profile(x)
    S_x = source_S(x)
    R = d2E_dx2 + (k0**2) * (n_x**2) * E - S_x
    return R

# =====================
# 7) Entraînement
# =====================

model = MLP(in_dim=1, out_dim=1, hidden=5, width=128, act=nn.Tanh).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

x_int_t = x_int.clone().detach()
x_bc_t  = x_bc.clone().detach()

# Valeurs cibles aux bords (Dirichlet)
E_bc_target = torch.zeros_like(x_bc_t).to(device)

num_epochs = 8000
log_every = 500

loss_hist = []

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    # Résidu PDE intérieur
    R = pde_residual(model, x_int_t)
    loss_pde = torch.mean(R**2)

    # Perte de bord (Dirichlet E=0 aux extrémités)
    E_bc_pred = model(x_bc_t)
    loss_bc = torch.mean((E_bc_pred - E_bc_target)**2)

    # Normalisation : optionnel, éviter solution triviale lorsque S est très faible
    # On peut fixer E(0) ≈ 1 en ajoutant un point de contrainte.
    x_ctr = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    E_ctr = model(x_ctr)
    loss_ctr = (E_ctr - 1.0)**2

    loss = loss_pde + 1.0 * loss_bc + 0.01 * loss_ctr
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_hist.append([epoch, loss.item(), loss_pde.item(), loss_bc.item(), loss_ctr.item()])

    if epoch % log_every == 0:
        print(f"Epoch {epoch:5d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e} | bc={loss_bc.item():.3e} | ctr={loss_ctr.item():.3e}")

# =====================
# 8) Post-traitement et visualisation
# =====================

model.eval()
x_plot = torch.linspace(-L, L, 1200, dtype=torch.float32).view(-1, 1).to(device)
with torch.no_grad():
    E_pred = model(x_plot).cpu().numpy().flatten()

x_plot_np = x_plot.cpu().numpy().flatten()
n_plot = n_profile(x_plot).cpu().numpy().flatten()
S_plot = source_S(x_plot).cpu().numpy().flatten()

# Plot champ vs indice
plt.figure(figsize=(7,4))
plt.title("Champ E(x) prédit par PINN et profil d'indice")
plt.plot(x_plot_np, E_pred, label="E(x) – PINN")
plt.plot(x_plot_np, (n_plot - n_clad) / (n_core - n_clad + 1e-9) * np.max(np.abs(E_pred))*0.8, \
         linestyle='--', label="n(x) (échelle relative)")
plt.xlabel("x [µm]")
plt.legend()
plt.tight_layout()
plt.show()

# Plot source
plt.figure(figsize=(7,3.5))
plt.title("Source gaussienne S(x)")
plt.plot(x_plot_np, S_plot, label="S(x)")
plt.xlabel("x [µm]")
plt.legend()
plt.tight_layout()
plt.show()

# Historique des pertes
import pandas as pd
loss_df = pd.DataFrame(loss_hist, columns=["epoch","loss","loss_pde","loss_bc","loss_ctr"])
plt.figure(figsize=(7,3.5))
plt.title("Historique des pertes")
plt.semilogy(loss_df["epoch"], loss_df["loss"], label="total")
plt.semilogy(loss_df["epoch"], loss_df["loss_pde"], label="pde")
plt.semilogy(loss_df["epoch"], loss_df["loss_bc"], label="bc")
plt.semilogy(loss_df["epoch"], loss_df["loss_ctr"], label="ctr")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# =====================
# 9) Extensions possibles (guides succincts)
# =====================
"""
1) Conditions aux limites absorbantes (Robin, 1ère ordre) :
   A x=±L, imposer dE/dx ± i*k0*n_clad*E = 0.
   -> Il faut passer le modèle en complexe (séparer réel/imag) et pénaliser
      ces conditions dans la loss avec autograd.

2) Champs complexes :
   - Sortie du réseau : 2 canaux (E_re, E_im). Calculer la PDE sur E = E_re + i E_im.
   - Les pertes deviennent la MSE de la partie réelle et imaginaire des résidus.

3) Profils n(x) plus réalistes :
   - Gradient d'indice (profil gaussien), multi-couches, défauts localisés.

4) 2D (Helmholtz ou Maxwell quasi-TE/TM) :
   - Entrée réseau (x,y), PDE avec ΔE + k0^2 n(x,y)^2 E = S(x,y).
   - Collocation en disque/carré, bords absorbants ou PML approximée.

5) Problème d'eigen-modes (n_eff) :
   - Reformuler en problème aux valeurs propres, ajouter une inconnue n_eff et
     normalisation d'énergie ; ou utiliser une boucle sur n_eff et une contrainte
     d'orthogonalité. Plus avancé mais faisable avec PINN.

6) Validation par FDTD/FDFD :
   - Comparer la solution PINN à une référence numérique (Meep, FEniCS, FDFD).
"""
