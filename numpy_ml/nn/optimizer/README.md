$$
\begin{align}
\text{update}^{(t)}
                &=  \text{momentum} \cdot \text{update}^{(t-1)} + \eta^{(t)} \nabla_{\theta} \mathcal{L}\\
            \theta^{(t+1)}
                &\leftarrow  \theta^{(t)} - \text{update}^{(t)}
\end{align}
$$