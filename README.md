# Non-Linear

A hand-curated collections of activations functions for deep learning research.

## Channel-Based Activation Functions

<div align="center">

| ID | Activation Function | Formula |
|---:|---|---|
| 1 | [DPReLU](nonlinear/channel_based/DPReLU.py) | $f(x)=ax$ if $x\ge0$, else $bx$ |
| 2 | [DualLine](nonlinear/channel_based/DualLine.py) | $f(x)=ax+m$ if $x\ge0$, else $bx+m$ |
| 3 | [EPReLU](nonlinear/channel_based/EPReLU.py) | $f(x)=kx$ if $x\ge0$, else $\dfrac{x}{a}$ |
| 4 | [FPAF](nonlinear/channel_based/FPAF.py) | $f(x)=a\mu(x)$ if $x\ge0$, else $b\nu(x)$ |
| 5 | [FReLU](nonlinear/channel_based/FReLU.py) | $f(x)=\mathrm{ReLU}(x)+b$ |
| 6 | [LearnableTeLU](nonlinear/channel_based/LearnableTeLU.py) | $f(x)=x\tanh(\mathrm{ELU}(ax))$ |
| 7 | [LeLeLU](nonlinear/channel_based/LeLeLU.py) | $f(x)=ax$ if $x\ge0$, else $0.01ax$ |
| 8 | [PairedReLU](nonlinear/channel_based/PairedReLU.py) | $f(x)=\left[\mathrm{ReLU}(ax-b),\mathrm{ReLU}(cx-d)\right]$ (concat on channels) |
| 9 | [PiLU](nonlinear/channel_based/PiLU.py) | $f(x)=ax+c(1-a)$ if $x\ge c$, else $bx+c(1-b)$ |
| 10 | [PREU](nonlinear/channel_based/PREU.py) | $f(x)=ax$ if $x\ge0$, else $ax\,e^{bx}$ |
| 11 | [PTELU](nonlinear/channel_based/PTELU.py) | $f(x)=x$ if $x\ge0$, else $\lvert a\rvert\,\tanh(\lvert b\rvert x)$ |
| 12 | [RMAF](nonlinear/channel_based/RMAF.py) | $f(x)=\dfrac{abx}{\left(0.25(1+e^{-x})+0.75\right)^c}$ |
| 13 | [RTPReLU](nonlinear/channel_based/RTPReLU.py) | $f(x)=x$ if $x+\eta\ge0$, else $x/a$; $\eta\sim\mathcal{N}(0,\sigma^2)$ during training |
| 14 | [ShiLU](nonlinear/channel_based/ShiLU.py) | $f(x)=a\,\mathrm{ReLU}(x)+b$ |
| 15 | [StarReLU](nonlinear/channel_based/StarReLU.py) | $f(x)=a\,\mathrm{ReLU}(x)^2+b$ |
| 16 | [TaLU](nonlinear/channel_based/TaLU.py) | $f(x)=x$ if $x\ge\lvert b\rvert$; $\tanh(x)$ if $x>\lvert a\rvert$; else $\tanh(\lvert a\rvert)$ |
| 17 | [TanhLU](nonlinear/channel_based/TanhLU.py) | $f(x)=a\,\tanh(cx)+bx$ |

</div>

## Static Activation Functions

<div align="center">

| ID | Activation Function | Formula |
|---:|---|---|
| 1 | [ShiftedReLU](nonlinear/static/ShiftedReLU.py) | $f(x)=\max(x,-1)$ |
| 2 | [ADA](nonlinear/static/ADA.py) | $f(x)=x$ if $x\ge0$, else $xe^x$ |
| 3 | [OAF](nonlinear/static/OAF.py) | $f(x)=\mathrm{ReLU}(x)+x\sigma(x)$ |
| 4 | [AbsLU](nonlinear/static/AbsLU.py) | $f(x)=x$ if $x\ge0$, else $\alpha\,\mathrm{abs}(x)$ |
| 5 | [ParametricLogish](nonlinear/static/ParametricLogish.py) | $f(x)=\alpha x\log(1+\sigma(\beta x))$ |
| 6 | [ExpExpish](nonlinear/static/ExpExpish.py) | $f(x)=xe^{-e^{-x}}$ |
| 7 | [DoubleSiLU](nonlinear/static/DoubleSiLU.py) | $f(x)=\dfrac{x}{1+\exp\!\left(-\dfrac{x}{1+e^{-x}}\right)}$ |
| 8 | [GeneralizedSwish](nonlinear/static/GeneralizedSwish.py) | $f(x)=x\,\sigma(e^{-x})$ |
| 9 | [MSiLU](nonlinear/static/MSiLU.py) | $f(x)=x\sigma(x)+\dfrac{1}{4}e^{-x^2-1}$ |
| 10 | [TBSReLU](nonlinear/static/TBSReLU.py) | $f(x)=x\tanh\!\left(\dfrac{1-e^{-x}}{1+e^{-x}}\right)$ |
| 11 | [ASiLU](nonlinear/static/ASiLU.py) | $f(x)=\arctan\!\left(\dfrac{x}{1+e^{-x}}\right)$ |
| 12 | [NoisyReLU](nonlinear/static/NoisyReLU.py) | $f(x)=x+\epsilon\,\mathrm{std}(x)$ if $x\ge0$, else $0$; $\epsilon\sim\mathcal{N}(0,1)$ |
| 13 | [ExponentialDLReLU](nonlinear/static/ExponentialDLReLU.py) | $f(x)=x$ if $x\ge0$, else $(ae^{-b_t})x$ |
| 14 | [SaRa](nonlinear/static/SaRa.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1+\alpha e^{-\beta x}}$ |
| 15 | [SiELU](nonlinear/static/SiELU.py) | $f(x)=x\,\sigma\!\left(2\sqrt{2/\pi}(x+0.044715x^3)\right)$ |
| 16 | [EANAF](nonlinear/static/EANAF.py) | $f(x)=x\tanh(\mathrm{softplus}(x)/2)$ |
| 17 | [MaxSig](nonlinear/static/MaxSig.py) | $f(x)=\max(x,\sigma(x))$ |
| 18 | [TangentSigmoidReLU](nonlinear/static/TangentSigmoidReLU.py) | $f(x)=x\tanh(\sigma(x))$ |
| 19 | [Phish](nonlinear/static/Phish.py) | $f(x)=x\tanh(\mathrm{GELU}(x))$ |
| 20 | [SelfArctan](nonlinear/static/SelfArctan.py) | $f(x)=x\arctan(x)$ |
| 21 | [PFLU](nonlinear/static/PFLU.py) | $f(x)=\dfrac{x}{2}\left(1+\dfrac{x}{\sqrt{1+x^2}}\right)$ |
| 22 | [ReSP](nonlinear/static/ReSP.py) | $f(x)=\alpha x+\log 2$ if $x\ge0$, else $\log(1+e^x)$ |
| 23 | [Serf](nonlinear/static/Serf.py) | $f(x)=x\,\mathrm{erf}(\log(1+e^x))$ |
| 24 | [LogSigmoid](nonlinear/static/LogSigmoid.py) | $f(x)=\log(\sigma(x))$ |
| 25 | [SlopedReLU](nonlinear/static/SlopedReLU.py) | $f(x)=\alpha x$ if $x\ge0$, else $0$ |
| 26 | [ReCU](nonlinear/static/ReCU.py) | $f(x)=\mathrm{ReLU}(x^3)$ |
| 27 | [MinSin](nonlinear/static/MinSin.py) | $f(x)=\min(x,\sin x)$ |
| 28 | [LaLU](nonlinear/static/LaLU.py) | $f(x)=x(1-0.5e^{-x})$ if $x\ge0$, else $0.5xe^x$ |
| 29 | [mReLU](nonlinear/static/mReLU.py) | $f(x)=\min(\mathrm{ReLU}(1-x),\mathrm{ReLU}(1+x))$ |
| 30 | [FlattedTSwish](nonlinear/static/FlattedTSwish.py) | $f(x)=\mathrm{ReLU}(x)\sigma(x)+t$ |
| 31 | [ERF](nonlinear/static/ERF.py) | $f(x)=x\,\mathrm{erf}(\alpha x)$ |
| 32 | [RePU](nonlinear/static/RePU.py) | $f(x)=\mathrm{ReLU}(x^{\alpha})$ |
| 33 | [TangentBipolarSigmoidReLU](nonlinear/static/TangentBipolarSigmoidReLU.py) | $f(x)=x\tanh\!\left(\dfrac{1-e^{-x}}{1+e^{-x}}\right)$ |
| 34 | [BaseDLReLU](nonlinear/static/BaseDLReLU.py) | $f(x)=x$ if $x\ge0$, else $sx$; $s=ab_t$ (linear) or $s=ae^{-b_t}$ (exp) |
| 35 | [Logish](nonlinear/static/Logish.py) | $f(x)=x\log(1+\sigma(x))$ |
| 36 | [TripleStateSwish](nonlinear/static/TripleStateSwish.py) | $f(x)=x\sigma(x)[\sigma(x)+\sigma(x-\alpha)+\sigma(x-\beta)]$ |
| 37 | [ReQU](nonlinear/static/ReQU.py) | $f(x)=\mathrm{ReLU}(x^2)$ |
| 38 | [ExponentialSwish](nonlinear/static/ExponentialSwish.py) | $f(x)=e^{-x}\sigma(x)$ |
| 39 | [SinSig](nonlinear/static/SinSig.py) | $f(x)=x\sin\!\left(\dfrac{\pi}{2}\sigma(x)\right)$ |
| 40 | [PLAF](nonlinear/static/PLAF.py) | $f(x)=x-(1-\dfrac{1}{d})$ if $x\ge1$; $-x-(1-\dfrac{1}{d})$ if $x<-1$; else $\dfrac{1}{d}(\mathrm{abs}(x))^d$ |
| 41 | [TeLU](nonlinear/static/TeLU.py) | $f(x)=x\tanh(e^x)$ |
| 42 | [DiffELU](nonlinear/static/DiffELU.py) | $f(x)=x$ if $x\ge0$, else $a(xe^x-be^{bx})$ |
| 43 | [Elliot](nonlinear/static/Elliot.py) | $f(x)=0.5+\dfrac{0.5x}{1+\mathrm{abs}(x)}$ |
| 44 | [SoftModulusQ](nonlinear/static/SoftModulusQ.py) | $f(x)=x^2(2-\mathrm{abs}(x))$ if $\mathrm{abs}(x)\le1$, else $\mathrm{abs}(x)$ |
| 45 | [DerivativeSiLU](nonlinear/static/DerivativeSiLU.py) | $f(x)=\sigma(x)\left(1+x(1-\sigma(x))\right)$ |
| 46 | [NLReLU](nonlinear/static/NLReLU.py) | $f(x)=\log(\beta\max(0,x)+1)$ |
| 47 | [IpLU](nonlinear/static/IpLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1+(\mathrm{abs}(x))^{\alpha}}$ |
| 48 | [ThLU](nonlinear/static/ThLU.py) | $f(x)=x$ if $x\ge0$, else $\tanh(x/2)$ |
| 49 | [RReLU](nonlinear/static/RReLU.py) | $f(x)=x$ if $x\ge0$, else $x/a$ with $a\sim U(l,u)$ during training |
| 50 | [PolyLU](nonlinear/static/PolyLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{1}{1-x}-1$ |
| 51 | [Suish](nonlinear/static/Suish.py) | $f(x)=\max(x,xe^{-\mathrm{abs}(x)})$ |
| 52 | [TSiLU](nonlinear/static/TSiLU.py) | $\alpha=\dfrac{x}{1+e^{-x}},\;f(x)=\dfrac{e^{\alpha}-e^{-\alpha}}{2e^{\alpha}}$ |
| 53 | [SoftsignRReLU](nonlinear/static/SoftsignRReLU.py) | $f(x)=\dfrac{1}{(1+x)^2}+x$ if $x\ge0$, else $\dfrac{1}{(1+x)^2}+ax$ with $a\sim U(l,u)$ |
| 54 | [SoftModulusT](nonlinear/static/SoftModulusT.py) | $f(x)=x\tanh(x/\alpha)$ |
| 55 | [Gish](nonlinear/static/Gish.py) | $f(x)=x\log(2-e^{-e^x})$ |
| 56 | [DRLU](nonlinear/static/DRLU.py) | $f(x)=\mathrm{ReLU}(x-\alpha)$ |
| 57 | [NReLU](nonlinear/static/NReLU.py) | $f(x)=x+\epsilon\,\mathrm{std}(x)$ if $x\ge0$, else $0$; $\epsilon\sim\mathcal{N}(0,1)$ |
| 58 | [LogLogish](nonlinear/static/LogLogish.py) | $f(x)=x(1-e^{-e^x})$ |
| 59 | [SGELU](nonlinear/static/SGELU.py) | $f(x)=\alpha x\,\mathrm{erf}(x/\sqrt{2})$ |
| 60 | [TSReLU](nonlinear/static/TSReLU.py) | $f(x)=x\tanh(\sigma(x))$ |
| 61 | [REU](nonlinear/static/REU.py) | $f(x)=x$ if $x\ge0$, else $xe^x$ |
| 62 | [ReSech](nonlinear/static/ReSech.py) | $f(x)=x\,\mathrm{sech}(x)=\dfrac{2x}{e^x+e^{-x}}$ |
| 63 | [SineReLU](nonlinear/static/SineReLU.py) | $f(x)=x$ if $x\ge0$, else $\epsilon(\sin x-\cos x)$ |
| 64 | [DLReLU](nonlinear/static/DLReLU.py) | $f(x)=x$ if $x\ge0$, else $(ab_t)x$ |
| 65 | [DLU](nonlinear/static/DLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{x}{1-x}$ |
| 66 | [CaLU](nonlinear/static/CaLU.py) | $f(x)=x\left(\dfrac{\arctan(x)}{\pi}+b\right)$ |
| 67 | [RandomizedSlopedReLU](nonlinear/static/RandomizedSlopedReLU.py) | $f(x)=\alpha x$ if $x\ge0$, else $0$; $\alpha\sim U(1,10)$ at init |
| 68 | [TanhExp](nonlinear/static/TanhExp.py) | $f(x)=x\tanh(e^x)$ |
| 69 | [PoLU](nonlinear/static/PoLU.py) | $f(x)=x$ if $x\ge0$, else $(1-x)^{-\alpha}-1$ |
| 70 | [GCU](nonlinear/static/GCU.py) | $f(x)=x\cos x$ |
| 71 | [SigmoidDerivative](nonlinear/static/SigmoidDerivative.py) | $f(x)=e^{-x}\sigma(x)^2$ |
| 72 | [Smish](nonlinear/static/Smish.py) | $f(x)=x\tanh(\log(1+\sigma(x)))$ |
| 73 | [SigLU](nonlinear/static/SigLU.py) | $f(x)=x$ if $x\ge0$, else $\dfrac{1-e^{-2x}}{1+e^{-2x}}$ |
| 74 | [AOAF](nonlinear/static/AOAF.py) | $f(x)=\mathrm{ReLU}(x-b\bar{x})+c\bar{x}$, $\bar{x}$ is channel mean over batch/spatial dims |

</div>