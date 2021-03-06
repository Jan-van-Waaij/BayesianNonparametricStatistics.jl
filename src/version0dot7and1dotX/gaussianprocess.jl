# gaussianprocess.jl

"""
    abstract type AbstractGaussianProcess end

Subtypes: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
Supertype: Any

Both subtypes represent a Gaussian process as a series expansion
in a basis of functions, where the vector of functions is multivariate normally
distributed. 

See also: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
"""
abstract type AbstractGaussianProcess end

"""
  struct GaussianProcess{S, T} <: AbstractGaussianProcess where {
      S<:AbstractVector{U} where U<:Function, T<:Union{AbstractMvNormal,GaussianVector}}
    basis::S
    distribution::T
  end

Implements a Gaussian process defined as an inner product of a Gaussian vector
with distribution 'distribution' of type GaussianVector or a subtype of AbstractMvNormal 
(from the Distributions.jl package) and a vector of functions,
the basis of the function space. For the use of a Gaussian process expanded in 
Faber-Schauder functions up to level j, we recommend using 
FaberSchauderExpansionWithGaussianCoefficients, as this makes efficiently use of
the sparsity structure of Faber-Schauder functions.

See also: FaberSchauderExpansionWithGaussianCoefficients, AbstractGaussianProcess

# Example

```julia
using Distributions
distribution = MvNormal([k^(-1.0) for k in 1:10])
Π = GaussianProcess([fourier(k) for k in 1:10], distribution)
f = rand(Π)
```
"""
struct GaussianProcess{S, T} <: AbstractGaussianProcess where {
    S<:AbstractVector{U} where U<:Function, T<:Union{AbstractMvNormal,GaussianVector}}
  basis::S
  distribution::T
  function GaussianProcess(basis::S, distribution::T) where {
        S<:AbstractArray{<:Function}, T<:Union{AbstractMvNormal,GaussianVector}}
      length(basis)==length(distribution) || throw(DimensionMismatch(
        "The basis and distribution are not of equal length."))
      new{S, T}(basis, distribution)
  end
end

# Tests need to be written. 
"""
    calculateboundssupport(higestlevel::Int)

Internal function, not exported. 

Returns a tuple of two Float64 vectors of length 2^(higestlevel+1)
where the i-th element of the first vector is the left bound of the 
support of the i-th Faber-Schauder basis function and the i-th 
element of the second vector is the right bound of the support of 
i-th Faber-Schauder basis function.
"""
function calculateboundssupport(higestlevel::Int)
    higestlevel ≥ 0 || throw(AssertionError("higestlevel should be positive."))
    lengthvector = 2^(higestlevel+1)
    leftboundsupport = Vector{Float64}(undef, lengthvector)
    rightboundsupport = Vector{Float64}(undef, lengthvector)
    leftboundsupport[1] = leftboundsupport[2] = 0.0
    rightboundsupport[1] = rightboundsupport[2] = 1.0
    for j in 1:higestlevel
        for k in 1:2^j
            i = 2^j+k
            leftboundsupport[i] = ldexp(float(k-1), -j)
            rightboundsupport[i] = ldexp(float(k), -j)
        end
    end
    return (leftboundsupport, rightboundsupport)
end

"""
    createFaberSchauderBasisUpToLevelHigestLevel(higestlevel::Int)

Internal function, not exported.

Creates Faber-Schauder basis up to level higestlevel. Returns a 
Vector{Function} type of length 2^(higestlevel+1). The first element 
of the vector is faberschauderone, and the 2^j+k element is 
faberschauder(j,k), with 0≤j≤higestlevel and 1≤k≤2^j. 
"""
function createFaberSchauderBasisUpToLevelHigestLevel(higestlevel::Int)
    higestlevel≥0 || throw(AssertionError("The levels are non-negative."))
    # There are 2^(higestlevel+1) basis functions. 
    basis = Vector{Function}(undef, 2^(higestlevel+1))
    basis[1] = faberschauderone
    for j in 0:higestlevel
        for k in 1:2^j 
            basis[2^j+k] = faberschauder(j,k)
        end
    end 
    return basis 
end 

"""
    struct FaberSchauderExpansionWithGaussianCoefficients{T} <:
            AbstractGaussianProcess where T<:Union{AbstractMvNormal,GaussianVector}
        higestlevel::Int64
        basis::Vector{Function}
        leftboundssupport::Vector{Float64}
        rightboundssupport::Vector{Float64}
        distribution::T
    end

Implements a Gaussian Process with Faber-Schauder functions and optimally exploits
the sparsity structure of Faber-Schauder functions.

Constructors:
    FaberSchauderExpansionWithGaussianCoefficients(higestlevel, distribution)
    FaberSchauderExpansionWithGaussianCoefficients(standarddeviationsperlevel::Vector{Float64})

The user is not allowed to set a basis, which the constructor calculates from the input and 
is by its very definition a Faber-Schauder basis. The length of distribution 
should be equal to 2^(higestlevel+1). The second constructor defines a Gaussian 
process with Faber-Schauder basis with independent coefficients with 
length(standarddeviationsperlevel)-1 levels (there is also a level zero), so 
2^(length(standarddeviationsperlevel)) number of basis functions, where the standard deviation 
of the coefficients belonging to level k is standarddeviationsperlevel[k+1].

# Example

```julia
α = 0.5
Π = FaberSchauderExpansionWithGaussianCoefficients([2^(-j*α) for j in 0:5])
f = rand(Π)
```
"""
struct FaberSchauderExpansionWithGaussianCoefficients{T} <:
         AbstractGaussianProcess where T<:Union{AbstractMvNormal, GaussianVector}
     higestlevel::Int64
     basis::Vector{Function}
     leftboundssupport::Vector{Float64}
     rightboundssupport::Vector{Float64}
     distribution::T
     function FaberSchauderExpansionWithGaussianCoefficients(higestlevel::Int64,
             distribution::T) where {T<:Union{AbstractMvNormal,GaussianVector}}
         higestlevel ≥ 0 || throw(AssertionError("higestlevel should be non-negative."))
         length(distribution) == 2^(higestlevel+1) || throw(AssertionError("The length of the
         distribution is not equal to 2^(higestlevel+1)."))
         basis = createFaberSchauderBasisUpToLevelHigestLevel(higestlevel)
         leftboundssupport, rightboundssupport = calculateboundssupport(
             higestlevel)
         new{T}(higestlevel, basis,
             leftboundssupport, rightboundssupport, distribution)
     end
end

"""
    createvectorofstandarddeviationsfromstandarddeviationsperlevel(standarddeviationsperlevel::AbstractArray{Float64})

Internal function, not exported!

Calculated a vector of standard deviations for all basis functions, where basis function i gets standard deviation 
standarddeviationsperlevel[ℓ+1] if ℓ is the level of i (we start at level 0).
"""
function createvectorofstandarddeviationsfromstandarddeviationsperlevel(standarddeviationsperlevel::AbstractArray{Float64})
    lenghtstandarddeviationsperlevel = length(standarddeviationsperlevel) 
    lenghtstandarddeviationsperlevel > 0 || throw(AssertionError("standarddeviationsperlevel is of zero length."))
    # The distribution has length 2^lenghtstandarddeviationsperlevel=2^(higestlevel+1).
    vectorofstandarddeviations = Vector{Float64}(undef, 2^lenghtstandarddeviationsperlevel)
    # There are two functions of level zero.
    vectorofstandarddeviations[1:2] .= standarddeviationsperlevel[1]
    # and 2^k of level k, k=1,2,...
    for k in 1:lenghtstandarddeviationsperlevel-1
        vectorofstandarddeviations[2^k+1:2^(k+1)] .= standarddeviationsperlevel[k+1]
    end
    return vectorofstandarddeviations  
end 

# Constructor
function FaberSchauderExpansionWithGaussianCoefficients(
        standarddeviationsperlevel::AbstractArray{Float64})
    lenghtstandarddeviationsperlevel = length(standarddeviationsperlevel)
    lenghtstandarddeviationsperlevel == 0 && throw(AssertionError("standarddeviationsperlevel is of
        zero length."))
    # We start with level zero.
    higestlevel = lenghtstandarddeviationsperlevel - 1
    vectorofstandarddeviations = createvectorofstandarddeviationsfromstandarddeviationsperlevel(standarddeviationsperlevel)
    distribution = MvNormal(vectorofstandarddeviations)
    return FaberSchauderExpansionWithGaussianCoefficients(higestlevel,
        distribution)
end

#Extends Base.length to objects of a subtypes of AbstractGaussianProcess.
"""
    length(Π::AbstractGaussianProcess)
    length(Π::FaberSchauderExpansionWithGaussianCoefficients)

length returns the number of basis functions, which is equal to the length of the distribution of the
coefficients, which is also equal to the number of coefficients. 
For a FaberSchauderExpansionWithGaussianCoefficients object, the length is equal to 2^(higestlevel+1).

# Example
```julia
Π = GaussianProcess([sin, cos], MvNormal([1.,1.]))
length(Π)
#
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^(-0.5*j) for j in 0:3])
length(Π) # == 2^4 == 16.
```
"""
length(Π::AbstractGaussianProcess)=length(Π.basis)
length(Π::FaberSchauderExpansionWithGaussianCoefficients) = 2^(Π.higestlevel+1)

"""
    sumoffunctions(vectoroffunctions::AbstractArray{<:Function},
        vectorofscalars::AbstractArray{<:Number})

Internal function, not exported.

sumoffunctions calculates the 'inner product' of the function vector and the scalar vector. In
other words, the sum of the functions weighted by vectorofscalars. It returns a
function.
"""
function sumoffunctions(vectoroffunctions::AbstractArray{<:Function},
    vectorofscalars::AbstractArray{<:Number})
  n = length(vectoroffunctions)
  n == length(vectorofscalars) || throw(AssertionError("The vector of functions and the vector
    of scalars should be of equal length"))
  return x -> sum(vectorofscalars[k]*vectoroffunctions[k](x) for k in 1:n)
end

"""
    rand(Π::AbstractGaussianProcess)

rand returns a random function, where the coefficients have distribution Π.distribution
and the basis functions are defined in Π.basis.

# Example

```julia
using Distributions
distribution = MvNormal([k^(-1.0) for k in 1:100])
Π = GaussianProcess([fourier(k) for  k in 1:100], distribution)
f = rand(Π)
```
"""
function rand(Π::AbstractGaussianProcess)
  Z = rand(Π.distribution)
  return sumoffunctions(Π.basis, Z)
end

# Extends Base.mean to GaussianProcess objects.
"""
    mean(Π::AbstractGaussianProcess)

mean calculates the mean of the Gaussian process. It returns a function.

#Examples

```julia
using Distributions
distribution = MvNormal([k^(-1.0) for k in 1:100])
Π = GaussianProcess([fourier(k) for  k in 1:100], distribution)
mean(Π)
```
"""
mean(Π::AbstractGaussianProcess) = sumoffunctions(Π.basis, mean(Π.distribution))
