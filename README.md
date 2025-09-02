# Helmholtz-PINN

Un **Physics-Informed Neural Network (PINN)** pour résoudre l’équation de **Helmholtz**, utilisée dans la modélisation des ondes (acoustique, électromagnétisme, mécanique des vibrations).

---

## ⚙️ Formulation mathématique

L’équation de Helmholtz est donnée par :  

\[
\nabla^2 u(\mathbf{x}) + k^2 u(\mathbf{x}) = f(\mathbf{x}), \quad \mathbf{x} \in \Omega
\]

- \( u(\mathbf{x}) \) : fonction d’onde inconnue  
- \( k \) : nombre d’onde (lié à la fréquence \(\omega\) et la vitesse \(c\) par \(k = \omega/c\))  
- \( f(\mathbf{x}) \) : terme source  
- \( \Omega \) : domaine spatial  

### Conditions aux limites
- **Dirichlet** :  
\[
u(\mathbf{x}) = g(\mathbf{x}), \quad \mathbf{x} \in \partial \Omega
\]

- **Neumann** :  
\[
\frac{\partial u}{\partial n} (\mathbf{x}) = h(\mathbf{x}), \quad \mathbf{x} \in \partial \Omega
\]

---

## 🧠 Méthode PINN

Le réseau de neurones \( u_\theta(\mathbf{x}) \) approxime la solution.  
La **fonction de perte** combine :  

1. **Erreur résiduelle de l’équation de Helmholtz :**

\[
\mathcal{L}_{\text{PDE}} = \frac{1}{N_r} \sum_{i=1}^{N_r} 
\Big| \nabla^2 u_\theta(\mathbf{x}_i) + k^2 u_\theta(\mathbf{x}_i) - f(\mathbf{x}_i) \Big|^2
\]

2. **Erreur sur les conditions aux limites :**

\[
\mathcal{L}_{\text{BC}} = \frac{1}{N_b} \sum_{i=1}^{N_b} 
\Big| u_\theta(\mathbf{x}_i) - g(\mathbf{x}_i) \Big|^2
\]

3. **Perte totale :**

\[
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda \, \mathcal{L}_{\text{BC}}
\]

---

## 📂 Structure du dépôt

