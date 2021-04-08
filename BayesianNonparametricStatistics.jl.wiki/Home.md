BayesianNonparametricStatistics.jl unofficial Julia package wiki!

The BayesianNonparametricStatistics package provides functions to model SDE's and to sample from them. Methods are provided to sample from the posterior, or to calculate the posterior mean.

BayesianNonparametricStatistics works with Julia 0.6 and Julia 0.7. The right version of the software is automatically installed. The version for Julia 0.6 requires no other package, the version for Julia 0.7 requires StatsBase, LinearAlgebra and SparseArrays to be installed. 

# AbstractSamplePath

An abstract type. Has supertype Any and, so far, only SamplePath as subtype. The user might define its own AbstractSamplePath subtypes.

# Sample paths. 

Sample paths are stored in an efficient way in a SamplePath type, which is available to the user. SamplePath is essentially implemented as follows:

```julia
struct SamplePath{S, T} <: AbstractSamplePath where {
    S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}
  timeinterval::S
  samplevalues::T
end
``` 
timeinterval needs to be increasing. Use a range object for timeinterval whenever possible!

## Constructors:
* SamplePath(t,x), where t and x are of the same size (otherwise an error is thrown) and t is the grid of time points and x are the corresponding values, so the samplepath has value x[k] at time t[k]. t needs to be increasing, otherwise, an error is thrown.
* SamplePath(t, f), where t is again a time interval, but f is a function, evaluates to SamplePath(t, f.(t)), as above. So the value of the sample path at time t[k] is f(t[k]). f needs to be a function that takes Float64 and returns Float64. 

## Methods: 

* length(X::SamplePath), returns the length of X.timeinterval==X.samplevalues.
* step(X::SamplePath{S}) where S<:Range{Float64}, returns step(X.timeinterval), only if timeinterval is a Range object, otherwise a method error is thrown.

## Non-exported functios:

This function is used 'under the hood': 

### isincreasing(x)

Methods: 

```julia
isincreasing(x::S) where S<:AbstractArray{T} where T <: Number
isincreasing(x::S) where S<:Range{T} where T <: Number
```

tests wheter x is a strictly increasing function, so x[k+1]>x[k] for every k ∈ 1:length(x)-1. 

## Plotting of SamplePath objects.

Depending on your taste, you might use either PyPlot of Gadfly (or any other Julia plotting package). In the description below, X is a SamplePath object. 

### PyPlot
If you use PyPlot, type 
```julia
using PyPlot
plot(X.timeinterval, X.samplevalues)
```
or define a plot method for SamplePath objects:
```julia
import PyPlot.plot
plot(X::SamplePath)=plot(X.timeinterval, X.samplevalues)
```
and then you can use 
```julia
plot(X)
```

### Gadfly
If you use Gadfly, type 
```julia
using Gadfly
plot(x=X.timeinterval, y=X.samplevalues, Geom.line)
```
or define a plot method for SamplePath objects:
```julia
import Gadfly.plot
plot(X::SamplePath)=plot(x=X.timeinterval, y=X.samplevalues, Geom.line)
```
and then you can use 
```julia
plot(X)
```

## Example

The following plots the function sin(2πx) on [0,1] using SamplePath:
### (with PyPlot)
```julia
using BayesianNonparametricStatistics, PyPlot
clf()
timeinterval = 0.0:0.001:1.0
values = map(x->sinpi(2x), timeinterval)
X = SamplePath(timeinterval, values)
plot(X.timeinterval, X.samplevalues)
```
### (with Gadfly)
```julia
using BayesianNonparametricStatistics, Gadfly
timeinterval = 0.0:0.01:1.0
values = map(x->sinpi(2x), timeinterval)
X = SamplePath(timeinterval, values)
plot(x=X.timeinterval, y=X.samplevalues, Geom.line)
```

# Statistical models: AbstractModel and SDEModel.

The model for the data can be specified as a concrete subtype of AbstractModel; an abstract type that is available to the user.
 
```julia
abstract type AbstractModel end
```
AbstractModel has supertype Any and in the package, only SDEModel is defined as a subtype, which is basically implemented as follows:

```julia
struct SDEModel{T} <: AbstractModel where T<:Union{Float64, <:Function}
    σ::T
    beginvalue::Float64
    endtime::Float64
    Δ::Float64
end
```

It is a model for all diffusions X={X_t:t∈[0,endtime]} discretised with precision Δ, satisfying the SDE dX_t=b(X_t)dt+σ(X_t)dW_t, with beginvalue X_0=beginvalue, where σ is assumed to be known and might be a (mathematically speaking) a nonconstant function, which is implemented here as a subtype of Function (taking a Float64 and returning a Float64), or a constant, which is most efficiently implemented as a Float64 primitive type. σ is assumed to be time-independent and the laws of X, for the different b under consideration, are assumed to be equivalent. 

# Sampling from a stochastic differential equation (SDE). 

BayesianNonparametricStatistics.jl provides tools to sample from an SDE dX_t=b(X_t)+σ(X_t)dW_t, both b and σ are time-independent. First of all the model needs to be specified as in the previous section. 

The full SDE is defined by

```julia
struct SDE{S, T} <: AbstractSDE where {S<:Function, T<:SDEModel}
  b::S
  model::T
end
```

The base function Base.rand is extended to SDE objects and generates a SamplePath object with timeinterval 0.0:Δ:endtime, approximately satisfying the SDE (using the Euler–Maruyama method), with samplevalues[1]=beginvalue.

## Example 

The following plots a solution to dX_t=sin(2πX_t)dt+dW_t on the time interval [0,10]:
### (using PyPlot)
```julia
using BayesianNonparametricStatistics, PyPlot
clf()
model = SDEModel(1.0, 0.0, 10.0, 0.01)
sde = SDE(x->sinpi(2x), model)
X = rand(sde)
plot(X.timeinterval, X.samplevalues)
```
### (using Gadfly)
```julia
using BayesianNonparametricStatistics, Gadfly
model = SDEModel(1.0, 0.0, 10.0, 0.01)
sde = SDE(x->sinpi(2x), model)
X = rand(sde)
plot(x=X.timeinterval, y=X.samplevalues, Geom.line)
```

## AbstractSDE

The user might want to implement his or her own SDE as a subtype of AbstractSDE. 

```julia
abstract type AbstractSDE end
```

## Non-exported function:

This function is used 'under the hood': 

### calculatenextsamplevalue(prevXval, b, model, BMincrement)

Methods:

```julia
calculatenextsamplevalue(prevXval::Float64, b::Function,
    model::SDEModel{Float64}, BMincrement::Float64)
calculatenextsamplevalue(prevXval::Float64, b::Function,
    model::T, BMincrement::Float64) where T<:SDEModel{S} where S<:Function
```

Used in __rand(sde::SDE)__ to calculate the next sample value with the [Euler–Maruyama method](https://en.wikipedia.org/wiki/Euler–Maruyama_method).

# Gaussian vector

GaussianVector is essentially defined as 

```julia
struct GaussianVector{R<:AbstractArray{Float64, 1}, S<:AbstractArray{Float64}, 
    T<:AbstractArray{Float64}}
  mean::R
  Σ::S
  var::T
  length::Int
  numberofGaussians::Int
end 
```

and represents a d-variate normal vector (say d = length(mean)), which is defined as the sum 

```julia
mean + Σ * x, 
```

where the mean is the mean, Σ is a d×e matrix and x is an e-dimensional standard normal vector. Thus the covariance matrix of __mean + Σ * x__ is __var=Σ * Σ'__. Hence the distribution is non-degenerated if and only if var is invertible.

## Constructors

```julia
GaussianVector(mean::R, Σ::S) where {R<:AbstractArray{Float64, 1}, S<:AbstractArray{Float64}}
GaussianVector(Σ::T) where T<:AbstractArray{Float64} = GaussianVector(spzeros(size(Σ,1)), Σ)
```
The last function evaluates to __GaussianVector(spzeros(size(Σ,1)), Σ)__, a Gaussian vector with zero mean. 

## Methods

```julia
rand(d::GaussianVector) = d.mean + d.Σ * randn((d.numberofGaussians, 1))
mean(d::GaussianVector) = d.mean
var(d::GaussianVector) = d.var 
length(d::GaussianVector) = d.length
```


# Gaussian processes

In BayesianNonparametricStatistics.jl Gaussian processes are implemented as a random function expansion in a certain basis where the coefficients have a multivariate normal distribution (implemented as GaussianVector-type). There are two types to define Gaussian processes, GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients, both subtypes of AbstractGaussianProcess, the easiest of them is GaussianProcess. 

GaussianProcess is essentially defined as 

```julia
struct GaussianProcess{S, T} <: AbstractGaussianProcess where {
    S<:AbstractVector{U} where U<:Function, T<:GaussianVector}
  basis::S
  distribution::T
end
```

The basis and the length of the distribution need to be the same. The coefficients may be dependent, which need to be specified with distribution.

## Example 

```julia
using BayesianNonparametricStatistics
distribution = GaussianVector(diagm([k^(-1.0) for k in 1.0:10.0]))
Π = GaussianProcess([fourier(k) for k in 1:10], distribution)
f = rand(Π)
```

stores a Gaussian process as an expansion of ten Fourier basis functions, with independent Gaussian coefficients, with variances k^(-2.0) (and mean zero). 

```julia
f = rand(Π)
```

samples a random function from Π. The type of f is a subtype of Function. In this example, _rand(Π)_ returns a function which is a basis expansion ([fourier(k) for k in 1:10] in this example) where the vector of coefficients is a random draw from the 10-variate normal distribution of the coefficients (in this example). 

Thus 

```julia
using PyPlot
clf()
x = 0.0:0.01:1.0
for k in 1:5
  f = rand(Π)
  y = f.(x)
  plot(x,y)
end 
```

plots five independent draws from the Gaussian process Π. 

## Bases

Two bases are implemented: The Fourier basis and the Faber-Schauder basis. 

For any Int64 k ≥ 1, 

```julia
fourier(k)
```
returns a function which implements ϕ_k(x)=sqrt(2)sin((k+1)π*x), when k is odd, and ϕ_k(x)=sqrt(2)cos(k*x), when k is even. 

The Faber-Schauder basis is implemented with two methods: 

```julia
faberschauderone(x::Float64)
```

implements the first Faber-Schauder basis function ψ_1, defined by ψ_1(x)=1-2x for 0≤x≤1/2 and
-1+2x for 0.5≤x≤1, and is 1-periodically extended to all x∈R. This basis function may be used to implement periodic functions which are nonzero at 0.0 and 1.0. 

For Int64 j, k, j≥0 and 1≤k≤2^j,

```julia
faberschauder(j,k)
```

returns a function which implements the Faber-Schauder function ψ_{j,k} with index (j,k) defined defined on [0,1] as 2^(j+1)(x-(k-1)2^j) on
(k-1)2^j≤x≤(2k-1)2^(j+1) and 1 - 2^(j+1)(x-(2k-1)2^(j+1)) on
[(2k-1)2^(j+1), k2^j] and zero outside these intervals, and then one-periodically extended. We call j the level of Faber-Schauder basis function ψ_{j,k}. The level of ψ_1 is 0. Hence level zero has two basis functions, and level j, j≥1, has 2^(j+1) basis functions. 

For instance 

```julia
using BayesianNonparametricStatistics, PyPlot
clf() 
j=3
x=0.0:0.001:1.0
for k in 1:2^j
    y = faberschauder(j,k).(x)
    plot(x,y)
end
```

plots all Faber-Schauder functions of level 3.

### Warning

Note the difference between faberschauderone and faberschauder. The first takes Float64 and returns Float64, the second takes a pair j, k Int64, j≥0 and 1≤k≤2^j and returns a function, which in turn takes a Float64 and returns a Float64.

## FaberSchauderExpansionWithGaussianCoefficients

Clearly a Gaussian process with Faber-Schauder basis functions could be implemented with GaussianProcess, however, for posterior computations, GaussianProcess won't make use of the sparsity structure of Faber-Schauder functions. Therefore we recommend using a FaberSchauderExpansionWithGaussianCoefficients object when using a Faber-Schauder basis. 

The FaberSchauderExpansionWithGaussianCoefficients is essentially defined as follows

```julia
struct FaberSchauderExpansionWithGaussianCoefficients{T} <:
         AbstractGaussianProcess where T<:GaussianVector
  higestlevel::Int64
  basis::Vector{Function}
  leftboundssupport::Vector{Float64}
  rightboundssupport::Vector{Float64}
  distribution::T
end
```

However, the user is not allowed to set basis (which is the Faber-Schauder basis) and leftboundssupport and rightboundssupport, which is calculated with 

```julia
calculateboundssupport(higestlevel::Int)
```

(not exported). The length of the distribution needs to be equal to the number of basis functions, which is equal to 2^(higestlevel+1) (otherwise an error is thrown). FaberSchauderExpansionWithGaussianCoefficients uses all basis functions defined by faberschauderone and faberschauder(j,k), of level j up to and including level higestlevel ≥ 0 (otherwise an error is thrown). The basis function basis[k] has support in the interval [leftboundssupport[k], rightboundssupport[k]].  

### Constructors

```julia
FaberSchauderExpansionWithGaussianCoefficients(higestlevel::Int64,
             distribution::T) where {T<:GaussianVector}
```
The length of distribution (calculated with length(distribution)) needs to be equal to 2^(higestlevel+1). 

```julia
FaberSchauderExpansionWithGaussianCoefficients(
        standarddeviationsperlevel::AbstractArray{Float64})
```
This constructs a FaberSchauderExpansionWithGaussianCoefficients object with higestlevel length(inversevariancesperlevel)-1 (as we start with level 0), all coefficients are independent and have zero mean and standard deviation at level j equal to inversevariancesperlevel[j-1] (again, because we start at level zero).

## AbstractGaussianProcess 

AbstractGaussianProcess is an abstract type, has Any as supertype and GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients as subtypes. The user might want to define his or her own AbstractGaussianProcess subtypes.

```julia
abstract type AbstractGaussianProcess end
```

## Methods:

### length(Π::AbstractGaussianProcess)

Extends Base.length to AbstractGaussianProcess objects. Returns the length(Π.basis), which is equal to length(Π.distribution). 

### mean(Π::AbstractGaussianProcess)
```julia
mean(Π::AbstractGaussianProcess)
```
Extends Distributions.mean to AbstractGaussianProcess objects. Returns the mean of the Gaussian process, represented as a subtype of Function. 

### sumoffunctions(vectoroffunctions, vectorofscalars)

```julia
sumoffunctions(vectoroffunctions::S,
  vectorofscalars::T) where {S<:AbstractArray{U} where U<:Function,
  T<:AbstractArray{Float64}}
```

Returns the function x-> sum(vectorofscalars[k]*vectoroffunctions[k\](x) for k in 1:length(vectoroffunctions)).
Throws an error when the length of vectoroffunctions and vectorofscalars is not equal. 

# Calculate the posterior of a diffusion with a Gaussian process prior.

## Steps: 

1. Store your data in a SamplePath object _X_.
1. Describe the model as an SDEModel object _model_.
1. Store your prior as a GaussianProcess or FaberSchauderExpansionWithGaussianCoefficients object _Π_. 
1. Calculate the posterior with _postΠ = calculateposterior(Π, X, model)_. _calculateposterior_ stores the posterior as a Gaussian process, of the same type as _Π_.

_f = rand(postΠ)_ will draw a random function from the posterior. For example 

```julia
using PyPlot 
clf()
x = 0.0:0.001:1.0
y = f.(x)
plot(x,y)
``` 
will plot _f_. 

## Example with Faber-Schauder basis functions:

The following code generates data _X_ from the SDE dX_t=sin(2πX_t)dt+dW_t, then calculates the posterior, which is stored again as a Gaussian process, and then five draws from the posterior are drawn.

```julia
using BayesianNonparametricStatistics, PyPlot
clf()
model = SDEModel(1.0, 0.0, 10000.0, 0.01)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^(-j) for j in 0:5])
model = SDEModel(1.0, 0.0, 10.0, 0.01)
postΠ = calculateposterior(Π, X, model)
x=0.0:0.01:1.0
for k in 1:5
    f = rand(postΠ)
    plot(x, f.(x))
end
```

## Example with Fourier functions

The following code generates data _X_ from the SDE dX_t=sin(2πX_t)dt+dW_t, then calculates the posterior, which is stored again as a GaussianProcess object, and then five draws from the posterior as drawn.

```julia
using BayesianNonparametricStatistics, PyPlot
clf()
model = SDEModel(1.0, 0.0, 10000.0, 0.1)
sde = SDE(x->sinpi(2*x), model)
X = rand(sde)
distribution = GaussianVector(sparse(Diagonal([k^(-1.0) for k in 1.0:50.0])))
Π = GaussianProcess([fourier(k) for k in 1:50], distribution)
postΠ = calculateposterior(Π, X, model)
x=0.0:0.01:1.0
for k in 1:10
    f = rand(postΠ)
    plot(x, f.(x))
end
```

## Methods

```julia
calculateposterior(X::SamplePath, Π::GaussianProcess, model::SDEModel)
```

```julia
calculateposterior(X::SamplePath, 
  Π::FaberSchauderExpansionWithGaussianCoefficients, model::SDEModel)
```