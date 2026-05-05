import re

with open('main2.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Merge Sections V and VI
sec5_match = re.search(r'% ============================================================\n\\section\{Simulation Scenarios and Validation Strategy\}.*?(?=% ============================================================\n\\section\{GPU Programming and Performance\})', content, re.DOTALL)

if sec5_match:
    sec5_text = sec5_match.group(0)
    merged_text = r'''% ============================================================
\section{Simulation Scenarios and Validation Results}
% ============================================================

Three simulation cases are used to systematically verify the numerical
solver against the Savage--Hutter analytical solutions, each isolating a specific part of the coupled
mass--momentum system.

\subsection{Case 1: Mass Conservation Verification}

The analytical similarity velocity profile is prescribed while the
mass conservation equation is solved numerically to obtain (x,t)$.
This isolates the spatial discretization of the mass equation and
confirms correct spreading dynamics and mass preservation.
The numerical thickness profiles quickly relax into a parabolic shape.
The front and rear margins advance outward at nearly constant rates
for large $, matching the asymptotic spreading trend (t)\sim t$.
Small deviations near the boundaries arise from numerical diffusion
inherent to first-order upwinding.

\subsection{Case 2: Momentum Solver Verification}

The analytical thickness profile is imposed while the momentum
equation is solved numerically for (x,t)$.
This evaluates the accuracy of the momentum solver independently,
testing advection stability and internal-pressure effects.
The numerically computed velocity distribution maintains nearly linear
spatial variation (x,t)\propto\eta$, as predicted by the similarity
solution.
Minor phase lag between numerical and analytical profiles decreases
with grid refinement, confirming correct implementation of
internal-pressure and advective transport terms.

\subsection{Case 3: Fully Coupled Dynamics}

Both mass and momentum equations are solved simultaneously using the
finite difference scheme (Fig.~\ref{fig:case3_coupled}), representing
the full physical problem.
In the coupled case, the solution converges naturally toward the
parabolic cap profile even when initial conditions are not perfectly
parabolic, providing numerical confirmation of the theoretical
\emph{shape stability} of the solution.
The spreading rate and profile alignment with analytical similarity
solutions indicate strong accuracy of the coupled solver.

\begin{figure*}[t!]
    \centering
    \includegraphics[width=0.95\textwidth]{coupled.pdf}
    \caption{Case 3: Fully coupled numerical profiles of (x,t)$
    (left) and (x,t)$ (right) compared with the analytical
    Savage--Hutter parabolic cap similarity solution
    ($\zeta=32^\circ$, $\delta=22^\circ$, $\beta=0.204$).
    Note: The label `Lagrangian Numerical'' in the figure legend refers to the Eulerian finite-difference solver developed in this study.}
    \label{fig:case3_coupled}
\end{figure*}

This staged validation structure gives strong evidence that all solver
components are correct before GPU acceleration is introduced.
'''
    content = content.replace(sec5_text, merged_text)

# 2. Section reordering
gpu_2d_start = content.find('% ============================================================\n\\section{GPU Parallelisation of the Two-Dimensional Solver}')
ext_2d_start = content.find('% ============================================================\n\\section{Extension to Two-Dimensional Flow')
multi_gpu_start = content.find('% ============================================================\n\\section{Multi-GPU Performance Comparison}')

if gpu_2d_start != -1 and ext_2d_start != -1 and multi_gpu_start != -1:
    gpu_2d_block = content[gpu_2d_start:ext_2d_start]
    ext_2d_block = content[ext_2d_start:multi_gpu_start]
    
    # We swap them
    new_order = ext_2d_block + gpu_2d_block
    content = content[:gpu_2d_start] + new_order + content[multi_gpu_start:]

# 3. Eq. (2) friction sign switch
# Eq 2 is around line 192: 
content = content.replace(r'- \tan\delta\cos\zeta', r'- \tan\delta\cos\zeta\,\operatorname{sgn}(u)')
content = content.replace(r'is the resisting basal Coulomb friction (for\n>0$)', r'is the resisting basal Coulomb friction')

# 4. Hemispherical cap vs cylindrical cap
content = content.replace('reproduces a hemispherical cap of granular', 'reproduces a cylindrical cap of granular')

# 5. Shared-memory size
content = content.replace(r'($\approx4.6$\,KB for', r'($\approx9.0$\,KB for')

# 6. Conclusion ~5x FP32/FP64
content = content.replace(r'the $\sim5\times$', r'the .1\times$ (GTX~1650) and .0\times$ (A5000)')

# 7. Literature review
lit_review = r'''
\subsection{Literature Review}
Early depth-averaged models for granular avalanches were pioneered by Savage and Hutter \cite{Savage1989}, introducing the theoretical basis for shallow granular flows on inclined topographies. Extensions to complex geometries and varying friction laws were later developed by Pouliquen and Forterre \cite{Pouliquen2002}. Due to the high computational cost of solving these PDEs over large domains, recent efforts have shifted toward hardware acceleration. GPU-accelerated finite-volume solvers have emerged as the standard for geophysical hazard modeling, leveraging CUDA to achieve significant speedups over traditional CPU-based frameworks.
'''

if r'\section{Problem Definition}' in content:
    content = content.replace(r'\section{Problem Definition}', lit_review + '\n% ============================================================\n\\section{Problem Definition}')

# Add bibitems
bib_items = r'''\bibitem{Savage1989}
Savage, S. B., \& Hutter, K. (1989).
\textit{The motion of a finite mass of granular material down a rough incline}.
\textit{Journal of Fluid Mechanics}, 199, 177--215.

\bibitem{Pouliquen2002}
Pouliquen, O., \& Forterre, Y. (2002).
\textit{Friction law for dense granular flows: application to the motion of a mass down a rough inclined plane}.
\textit{Journal of Fluid Mechanics}, 453, 133--151.
'''

content = content.replace(r'\end{thebibliography}', bib_items + '\n\\end{thebibliography}')

with open('main2.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
