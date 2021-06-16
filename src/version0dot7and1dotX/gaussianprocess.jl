# gaussianprocess.jl

"""
    abstract type AbstractGaussianProcess end

Subtypes: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
Supertype: Any

Both subtypes represent a Gaussian process expressed as the inner product of a
Gaussian vector of real-valued coefficients and and a vector of functions. The
distribution is represented by a GaussianVector object.

See also: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
"""
abstract type AbstractGaussianProcess end

"""
  struct GaussianProcess{S, T} <: AbstractGaussianProcess where {
      S<:AbstractVector{U} where U<:Function, T<:GaussianVector}
    basis::S
    distribution::T
  end

Implements a Gaussian process defined as an inner product of a Gaussian vector
with distribution 'distribution' of type GaussianVector and a vector of functions,
the basis of the function space. For the use of a Gaussian process expanded in 
Faber-Schauder functions up to level j, we recommend using 
FaberSchauderExpansionWithGaussianCoefficients, as this makes efficient use of
the sparsity structure of Faber-Schauder functions.

See also: FaberSchauderExpansionWithGaussianCoefficients, AbstractGaussianProcess

# Example

```julia
distribution = GaussianVector(diagm(0 => [k^(-1.0) for k in 1.0:10.0]))
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
    struct FaberSchauderExpansionWithGaussianCoefficients{T} <:
            AbstractGaussianProcess where T<:GaussianVector
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

The user is not allowed to set basis, which is calculated from the input and 
is by its very definition a Faber-Schauder basis. The length of distribution 
should be equal to 2^(higestlevel+1). The second constructor defines a Gaussian 
process with Faber-Schauder basis with independent coefficients with 
length(standarddeviationsperlevel)-1 levels (there is also a level zero), so 
2^(length(standarddeviationsperlevel)) number of basis functions, where the variance 
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
         distribution.length == 2^(higestlevel+1) || throw(AssertionError("The length of the
         distribution is not equal to 2^(higestlevel+1)."))
         basis = vcat(faberschauderone, [faberschauder(j,k) for j in 0:higestlevel
             for k in 1:2^j])
         leftboundssupport, rightboundssupport = calculateboundssupport(
             higestlevel)
         new{T}(higestlevel, basis,
             leftboundssupport, rightboundssupport, distribution)
     end
end

# Constructor
function FaberSchauderExpansionWithGaussianCoefficients(
        standarddeviationsperlevel::AbstractArray{Float64})
    lenghtstandarddeviationsperlevel = length(standarddeviationsperlevel)
    lenghtstandarddeviationsperlevel == 0 && throw(AssertionError("standarddeviationsperlevel is of
        zero length."))
    # We start with level zero.
    higestlevel = lenghtstandarddeviationsperlevel - 1
    # There are two functions of level zero.
    vectorofstandarddeviations = repeat(standarddeviationsperlevel[1:1],2)
    # and 2^k of level k, k=1,2,...
    for k in 1:higestlevel
        vectorofstandarddeviations = vcat(vectorofstandarddeviations, repeat(
            standarddeviationsperlevel[k+1:k+1],2^k))
    end
#    distribution = GaussianVector(SparseArrays.sparse(LinearAlgebra.Diagonal(vectorofstandarddeviations)))
    distribution = MvNormal(vectorofstandarddeviations)
    return FaberSchauderExpansionWithGaussianCoefficients(higestlevel,
        distribution)
end

#Extends Base.length to objects of a subtypes of AbstractGaussianProcess.
"""
    length(Π::AbstractGaussianProcess)

Returns the number of basis functions == length of the distribution of the
coefficients. So for a FaberSchauderExpansionWithGaussianCoefficients object it
this is equal to 2^(higestlevel+1).

# Example
```julia
Π = GaussianProcess([sin, cos], MvNormal([1.,1.]))
length(Π)
#
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:3])
length(Π)
```
"""
length(Π::AbstractGaussianProcess)=length(Π.basis)

"""
    sumoffunctions(vectoroffunctions::Vector{<:Function},
        vectorofscalars::Vector{Float64})

Calculates the 'inner product' of the function vector and the scalar vector. In
other words the sum of the functions weigthed by vectorofscalars. Returns a
function.
"""
function sumoffunctions(vectoroffunctions::S,
    vectorofscalars::T) where {S<:AbstractArray{U} where U<:Function,
    T<:AbstractArray{Float64}}
  n = length(vectoroffunctions)
  n == length(vectorofscalars) || throw(AssertionError("The vector of functions and the vector
    of scalars should be of equal length"))
  return x -> sum(vectorofscalars[k]*vectoroffunctions[k](x) for k in 1:n)
end

# Extends Base.rand to objects which are subtypes of
# AbstractGaussianProcess. Returns a function.
"""
    rand(Π::AbstractGaussianProcess)

Returns a random function, where the coefficients have distribution Π.distribution
and the basis functions are defined in Π.basis.

# Example

```julia
distribution = GaussianVector([k^(-1.0) for k in 1:100])
Π = GaussianProcess([fourier(k) for  k in 1:100], distribution)
f = rand(Π)
```
"""
function rand(Π::AbstractGaussianProcess)
  Z = rand(Π.distribution)
  return sumoffunctions(Π.basis,Z)
end

# Extend Base.mean to GaussianProcess objects.
"""
    mean(Π::AbstractGaussianProcess)

Calculates the mean of the Gaussian process. Returns a function.

#Examples

```julia
distribution = GaussianVector([k^(-1.0) for k in 1:100])
Π = GaussianProcess([fourier(k) for  k in 1:100], distribution)
mean(Π)
```
"""
mean(Π::AbstractGaussianProcess) = sumoffunctions(Π.basis, mean(Π.distribution))
