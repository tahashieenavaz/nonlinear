# Non-Linear

A hand-curated collections of activations functions for deep learning research. 

## Channel-Based Activation Functions

<div align="center">

| Activation Function | Formula |
|---|---|
| [FPAF](nonlinear/channel_based/FPAF.py) | $f(x)=a\mu(x)$ if $x\ge0$, else $b\nu(x)$ |
| [DPReLU](nonlinear/channel_based/DPReLU.py) | $f(x)=ax$ if $x\ge0$, else $bx$ |
| [DualLine](nonlinear/channel_based/DualLine.py) | $f(x)=ax+m$ if $x\ge0$, else $bx+m$ |
| [FReLU](nonlinear/channel_based/FReLU.py) | $f(x)=\mathrm{ReLU}(x)+b$ |
| [LeLeLU](nonlinear/channel_based/LeLeLU.py) | $f(x)=ax$ if $x\ge0$, else $0.01ax$ |
| [PREU](nonlinear/channel_based/PREU.py) | $f(x)=ax$ if $x\ge0$, else $ax\,e^{bx}$ |
| [ShiLU](nonlinear/channel_based/ShiLU.py) | $f(x)=a\,\mathrm{ReLU}(x)+b$ |
| [StarReLU](nonlinear/channel_based/StarReLU.py) | $f(x)=a\,\mathrm{ReLU}(x)^2+b$ |
| [EPReLU](nonlinear/channel_based/EPReLU.py) | $f(x)=kx$ if $x\ge0$, else $\frac{x}{a}$ |
| [PairedReLU](nonlinear/channel_based/PairedReLU.py) | $f(x)=\left[\mathrm{ReLU}(ax-b),\mathrm{ReLU}(cx-d)\right]$ (concat on channels) |
| [RMAF](nonlinear/channel_based/RMAF.py) | $f(x)=\frac{abx}{\left(0.25(1+e^{-x})+0.75\right)^c}$ |
| [PTELU](nonlinear/channel_based/PTELU.py) | $f(x)=x$ if $x\ge0$, else $|a|\,\tanh(|b|x)$ |
| [TaLU](nonlinear/channel_based/TaLU.py) | $f(x)=x$ if $x\ge|b|$; $\tanh(x)$ if $x>|a|$; else $\tanh(|a|)$ |
| [TanhLU](nonlinear/channel_based/TanhLU.py) | $f(x)=a\,\tanh(cx)+bx$ |

</div>

## Static Activation Functions


| Activation Function | Formula |
|---|---|
| [ShiftedReLU](nonlinear/static/ShiftedReLU.py) | $f(x)=\max(x,-1)$ |
| [ADA](nonlinear/static/ADA.py) | $f(x)=x$ if $x\ge0$, else $xe^x$ |
| [OAF](nonlinear/static/OAF.py) | $f(x)=\mathrm{ReLU}(x)+x\sigma(x)$ |
| [AbsLU](nonlinear/static/AbsLU.py) | $f(x)=x$ if $x\ge0$, else $\alpha\,\mathrm{abs}(x)$ |
| [ParametricLogish](nonlinear/static/ParametricLogish.py) | $f(x)=\alpha x\log(1+\sigma(\beta x))$ |
| [ExpExpish](nonlinear/static/ExpExpish.py) | $f(x)=xe^{-e^{-x}}$ |
| [DoubleSiLU](nonlinear/static/DoubleSiLU.py) | $f(x)=\dfrac{x}{1+\exp\!\left(-\dfrac{x}{1+e^{-x}}\right)}$ |
| [GeneralizedSwish](nonlinear/static/GeneralizedSwish.py) | $f(x)=x\,\sigma(e^{-x})$ |
| [MSiLU](nonlinear/static/MSiLU.py) | $f(x)=x\sigma(x)+\dfrac{1}{4}e^{-x^2-1}$ |
| [TBSReLU](nonlinear/static/TBSReLU.py) | $f(x)=x\tanh\!\left(\dfrac{1-e^{-x}}{1+e^{-x}}\right)$ |
| [ASiLU](nonlinear/static/ASiLU.py) | $f(x)=\arctan\!\left(\dfrac{x}{1+e^{-x}}\right)$ |
| [NoisyReLU](nonlinear/static/NoisyReLU.py) | $f(x)=x+\epsilon\sigma_x$ if $x\ge0$, else $0$ ($\epsilon\sim\mathcal{N}(0,1)$) |
| [ExponentialDLReLU](nonlinear/static/ExponentialDLReLU.py) | $f(x)=x$ if $x\ge0$, else $(ae^{-b_t})x$ |
| [SaRa](nonlinear/static/SaRa.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1+\alpha e^{-\beta x}}$ |
| [SiELU](nonlinear/static/SiELU.py) | $f(x)=x\,\sigma\!\left(2\sqrt{2/\pi}(x+0.044715x^3)\right)$ |
| [EANAF](nonlinear/static/EANAF.py) | $f(x)=x\tanh(\mathrm{softplus}(x)/2)$ |
| [MaxSig](nonlinear/static/MaxSig.py) | $f(x)=\max(x,\sigma(x))$ |
| [TangentSigmoidReLU](nonlinear/static/TangentSigmoidReLU.py) | $f(x)=x\tanh(\sigma(x))$ |
| [Phish](nonlinear/static/Phish.py) | $f(x)=x\tanh(\mathrm{GELU}(x))$ |
| [SelfArctan](nonlinear/static/SelfArctan.py) | $f(x)=x\arctan(x)$ |
| [PFLU](nonlinear/static/PFLU.py) | $f(x)=\dfrac{x}{2}\left(1+\dfrac{x}{\sqrt{1+x^2}}\right)$ |
| [ReSP](nonlinear/static/ReSP.py) | $f(x)=\alpha x+\log 2$ if $x\ge0$, else $\log(1+e^x)$ |
| [Serf](nonlinear/static/Serf.py) | $f(x)=x\,\mathrm{erf}(\log(1+e^x))$ |
| [LogSigmoid](nonlinear/static/LogSigmoid.py) | $f(x)=\log(\sigma(x))$ |
| [SlopedReLU](nonlinear/static/SlopedReLU.py) | $f(x)=\alpha x$ if $x\ge0$, else $0$ |
| [ReCU](nonlinear/static/ReCU.py) | $f(x)=\mathrm{ReLU}(x^3)$ |
| [MinSin](nonlinear/static/MinSin.py) | $f(x)=\min(x,\sin x)$ |
| [LaLU](nonlinear/static/LaLU.py) | $f(x)=x(1-0.5e^{-x})$ if $x\ge0$, else $0.5xe^x$ |
| [mReLU](nonlinear/static/mReLU.py) | $f(x)=\min(\mathrm{ReLU}(1-x),\mathrm{ReLU}(1+x))$ |
| [FlattedTSwish](nonlinear/static/FlattedTSwish.py) | $f(x)=\mathrm{ReLU}(x)\sigma(x)+t$ |
| [ERF](nonlinear/static/ERF.py) | $f(x)=x\,\mathrm{erf}(\alpha x)$ |
| [RePU](nonlinear/static/RePU.py) | $f(x)=\mathrm{ReLU}(x^{\alpha})$ |
| [TangentBipolarSigmoidReLU](nonlinear/static/TangentBipolarSigmoidReLU.py) | $f(x)=x\tanh\!\left(\dfrac{1-e^{-x}}{1+e^{-x}}\right)$ |
| [BaseDLReLU](nonlinear/static/BaseDLReLU.py) | $f(x)=x$ if $x\ge0$, else $sx$; $s=ab_t$ (linear) or $s=ae^{-b_t}$ (exp) |
| [Logish](nonlinear/static/Logish.py) | $f(x)=x\log(1+\sigma(x))$ |
| [TripleStateSwish](nonlinear/static/TripleStateSwish.py) | $f(x)=x\sigma(x)[\sigma(x)+\sigma(x-\alpha)+\sigma(x-\beta)]$ |
| [ReQU](nonlinear/static/ReQU.py) | $f(x)=\mathrm{ReLU}(x^2)$ |
| [ExponentialSwish](nonlinear/static/ExponentialSwish.py) | $f(x)=e^{-x}\sigma(x)$ |
| [SinSig](nonlinear/static/SinSig.py) | $f(x)=x\sin\!\left(\frac{\pi}{2}\sigma(x)\right)$ |
| [PLAF](nonlinear/static/PLAF.py) | $f(x)=x-(1-\frac{1}{d})$ if $x\ge1$; $-x-(1-\frac{1}{d})$ if $x<-1$; else $\frac{1}{d}(\mathrm{abs}(x))^d$ |
| [TeLU](nonlinear/static/TeLU.py) | $f(x)=x\tanh(e^x)$ |
| [DiffELU](nonlinear/static/DiffELU.py) | $f(x)=x$ if $x\ge0$, else $a(xe^x-be^{bx})$ |
| [Elliot](nonlinear/static/Elliot.py) | $f(x)=0.5+\dfrac{0.5x}{1+\mathrm{abs}(x)}$ |
| [SoftModulusQ](nonlinear/static/SoftModulusQ.py) | $f(x)=x^2(2-\mathrm{abs}(x))$ if $\mathrm{abs}(x)\le1$, else $\mathrm{abs}(x)$ |
| [DerivativeSiLU](nonlinear/static/DerivativeSiLU.py) | $f(x)=\sigma(x)\left(1+x(1-\sigma(x))\right)$ |
| [NLReLU](nonlinear/static/NLReLU.py) | $f(x)=\log(\beta\max(0,x)+1)$ |
| [IpLU](nonlinear/static/IpLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1+(\mathrm{abs}(x))^{\alpha}}$ |
| [ThLU](nonlinear/static/ThLU.py) | $f(x)=x$ if $x\ge0$, else $\tanh(x/2)$ |
| [ActivationFunction](nonlinear/ActivationFunction.py) | Base class (no `forward` formula) |
| [RReLU](nonlinear/static/RReLU.py) | $f(x)=x$ if $x\ge0$, else $x/a$ with $a\sim U(l,u)$ during training |
| [PolyLU](nonlinear/static/PolyLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{1}{1-x}-1$ |
| [Suish](nonlinear/static/Suish.py) | $f(x)=\max(x,xe^{-\mathrm{abs}(x)})$ |
| [TSiLU](nonlinear/static/TSiLU.py) | $\alpha=\dfrac{x}{1+e^{-x}},\;f(x)=\dfrac{e^{\alpha}-e^{-\alpha}}{2e^{\alpha}}$ |
| [SoftsignRReLU](nonlinear/static/SoftsignRReLU.py) | $f(x)=\frac{1}{(1+x)^2}+x$ if $x\ge0$, else $\frac{1}{(1+x)^2}+ax$ with $a\sim U(l,u)$ |
| [SoftModulusT](nonlinear/static/SoftModulusT.py) | $f(x)=x\tanh(x/\alpha)$ |
| [Gish](nonlinear/static/Gish.py) | $f(x)=x\log(2-e^{-e^x})$ |
| [DRLU](nonlinear/static/DRLU.py) | $f(x)=\mathrm{ReLU}(x-\alpha)$ |
| [NReLU](nonlinear/static/NReLU.py) | $f(x)=x+\epsilon\sigma_x$ if $x\ge0$, else $0$ ($\epsilon\sim\mathcal{N}(0,1)$) |
| [LogLogish](nonlinear/static/LogLogish.py) | $f(x)=x(1-e^{-e^x})$ |
| [SGELU](nonlinear/static/SGELU.py) | $f(x)=\alpha x\,\mathrm{erf}(x/\sqrt{2})$ |
| [TSReLU](nonlinear/static/TSReLU.py) | $f(x)=x\tanh(\sigma(x))$ |
| [REU](nonlinear/static/REU.py) | $f(x)=x$ if $x\ge0$, else $xe^x$ |
| [ReSech](nonlinear/static/ReSech.py) | $f(x)=x\,\mathrm{sech}(x)=\dfrac{2x}{e^x+e^{-x}}$ |
| [SineReLU](nonlinear/static/SineReLU.py) | $f(x)=x$ if $x\ge0$, else $\epsilon(\sin x-\cos x)$ |
| [DLReLU](nonlinear/static/DLReLU.py) | $f(x)=x$ if $x\ge0$, else $(ab_t)x$ |
| [DLU](nonlinear/static/DLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1-x}$ |
| [CaLU](nonlinear/static/CaLU.py) | $f(x)=x\left(\dfrac{\arctan(x)}{\pi}+b\right)$ |
| [RandomizedSlopedReLU](nonlinear/static/RandomizedSlopedReLU.py) | $f(x)=\alpha x$ if $x\ge0$, else $0$; $\alpha\sim U(1,10)$ at init |
| [TanhExp](nonlinear/static/TanhExp.py) | $f(x)=x\tanh(e^x)$ |
| [PoLU](nonlinear/static/PoLU.py) | $f(x)=x$ if $x\ge0$, else $(1-x)^{-\alpha}-1$ |
| [GCU](nonlinear/static/GCU.py) | $f(x)=x\cos x$ |
| [SigmoidDerivative](nonlinear/static/SigmoidDerivative.py) | $f(x)=e^{-x}\sigma(x)^2$ |
| [Smish](nonlinear/static/Smish.py) | $f(x)=x\tanh(\log(1+\sigma(x)))$ |
| [SigLU](nonlinear/static/SigLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{1-e^{-2x}}{1+e^{-2x}}$ |
| [AOAF](nonlinear/static/AOAF.py) | $f(x)=\mathrm{ReLU}(x-b\bar{x})+c\bar{x}$, $\bar{x}$ is channel mean over batch/spatial dims |
