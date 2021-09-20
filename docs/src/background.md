# The bits of math and science behind it

You can find the behind the scene augmentation theory in this paper : [Automated Augmented Conjugate Inference for Non-conjugate Gaussian Process Models](https://arxiv.org/abs/2002.11451).

## Gaussian Processes

To quote [Wikipedia](https://en.wikipedia.org/wiki/Gaussian_process)
*"A Gaussian process is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution, i.e. every finite linear combination of them is normally distributed. The distribution of a Gaussian process is the joint distribution of all those (infinitely many) random variables, and as such, it is a distribution over functions with a continuous domain, e.g. time or space."*

For a detailed understanding of Gaussian processes, check the [wonderful book of Rasmussen and Williams](http://www.gaussianprocess.org/gpml/) and for a quick introduction, check [this tutorial by Zoubin Ghahramani](http://mlss2011.comp.nus.edu.sg/uploads/Site/lect1gp.pdf).

Gaussian Processes are extremely practical models since they are non-parametric and are Bayesian. However the basic model is limited to regression with Gaussian noise and does not scale very well to large datasets (>1000 samples). The Augmented Gaussian Processes solve both these problems by adding inducing points as well as transforming the likelihood to get efficient variational inference.


## Augmented Gaussian Processes

We are interested in models which consist of a GP prior on a latent function $f\sim \text{GP}(0,k)$, where $k$ is the kernel function and the data $y$ is connected to $f$ via a non-conjugate likelihood $p(y|f)$ . We now aim on finding an augmented representation of the model which renders the model conditionally conjugate. Let $\omega$ be potential augmentation, then the augmented joint distribution is

$$p(y,f,\omega) =p(y|f,\omega)p(\omega)p(f).$$

The original model can be restored by marginalizing $\omega$, i.e. $p(y,f) =\int p(y,f,\omega)d\omega$.

The  goal  is  to  find  an  augmentation $\omega$,  such  that  the  augmented  likelihood $p(y|f,\omega$) becomes conjugate to the prior distributions $p(f)$ and $p(\omega)$ and the expectations of the log complete conditional distributions $\log p(f|\omega,y)$ and $\log p(\omega|f,y)$ can be computed in closed-form.

#### How to find a suitable augmentation?

Many popular likelihood functions can be expressed as a scale mixture of Gaussians

$$p(y|f) =\int N(y;Bf,\text{diag}(\omega^{−1}))p(\omega)d\omega,$$

where $B$ is a matrix (Palmer et al., 2006).  This representation directly leads to the augmented likelihood $p(y|\omega,f) =N(y;Bf,\text{diag}(\omega^{−1}))$ which is conjugate in $f$, i.e. the posterior is again a Gaussian. I am currently working on a generalized  and automatic approach, which should be available during this year.


#### Inference in the augmented model
If we assume that the augmentation, discussed in the previous section, was successful and that we obtained an augmented model $p(y,f,\omega) = p(y|f,\omega)p(f)p(\omega)$ which is conditionally conjugate.
In a conditionally conjugate model variational inference is easy and block coordinate ascent updates can be computed in closed-form.
We follow as structured mean-field approach and assume a decoupling between the latent GP $f$ and the auxiliary variable $\omega$ in the variational distribution $q(f,\omega) = q(f) q(\omega)$.  We alternate between updating $q(f)$ and $q(\omega)$ by using the typical coordinate ascent (CAVI) updates building on expectations of the log complete conditionals.

The hyperparameter of the latent GP (e.g. length scale) are learned by optimizing the variational lower bound as function of the hyper parameters. We alternate between updating the variational parameters and the hyperparameters.

## Sparse Gaussian Processes
Direct inference for GPs has a cubic computational complexity $\mathcal{O}(N^3)$. To scale our model to big datasets we approximate the latent GP by a *sparse GP* building on *inducing points*. This reduces the complexity to $\mathcal{O}(M^3)$, where $M$ is the number of inducing points.
Using inducing points allows us to employ stochastic variational inference (SVI) that computes the updates based on mini-batches of the data.
