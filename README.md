BayesianNonparametricStatistics.jl is an unofficial Julia package to sample from the nonparametric posteriors. Implements Gaussian process priors, implement tools to sample from a stochastic differential equation (SDE) and to estimate the drift parameter.

# Installation:

This package works with Julia 0.6, 0.7 and any version 1.X.

When using Julia 0.6, execute the following code: 

```julia
Pkg.clone("https://github.com/Jan-van-Waaij/BayesianNonparametricStatistics.git", "BayesianNonparametricStatistics")
```

when using Julia 0.7, execute


```julia
using Pkg
Pkg.clone("https://github.com/Jan-van-Waaij/BayesianNonparametricStatistics.git", "BayesianNonparametricStatistics")
```

In Julia 1.X press ] and copy-paste
```julia
add BayesianNonparametricStatistics
```

## Any problems? Contact me!

<jvanwaaij@gmail.com>

# Usage

After installation, type the following in your Julia script, or in a Julia REPL. 

```julia
  using BayesianNonparametricStatistics
```

to use the package. 

## Example 

Sample from an SDE dX_t=sin(2\pi X_t)dt+dW_t: 

```julia
  using BayesianNonparametricStatistics, Plots
  # implement SDE dX_t=sin(2\pi X_t)dt+dW_t, 
  # starting at zero till time 1000.0, discretised 
  # with precision 0.01.
  model = SDEModel(1.0,0.0,1000.0,0.01)
  sde = SDE(x->sinpi(2*x),model)
  # Sample from sde.
  X = rand(sde)
  # Plot X. 
  plot(X.timeinterval, X.samplevalues)
```

To recover the drift function, using Gaussian process posterior:
(the code is for Julia 0.7 or 1.0. When using 0.6 leave the line "using LinearAlgebra, SparseArrays" out and replace "Diagonal" in the second line by "diagm") 

```julia
  using BayesianNonparametricStatistics, LinearAlgebra, SparseArrays, Plots
  distribution = GaussianVector(sparse(Diagonal([k^(-1.0) for k in 1.0:50.0])))
  Π = GaussianProcess([fourier(k) for k in 1:50], distribution)
  postΠ = calculateposterior(Π, X, model)
  # sample 10 times from posterior
  plot()
  x = 0.0:0.01:1.0
  for k in 1:10
    f = rand(postΠ)
    y = f.(x)
    plot!(x,y,show=true)
  end 
```

# Wiki

Go to the [Wiki](https://github.com/Jan-van-Waaij/BayesianNonparametricStatistics.jl/wiki).

# Website

https://github.com/Jan-van-Waaij/BayesianNonparametricStatistics

# License

The BayesianNonparametricStatistics.jl package is licensed under the MIT "Expat" License:

> Copyright (c) 2017-2021: Jan van Waaij.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
