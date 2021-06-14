import Base: length, step, rand, iterate, eltype, getindex, firstindex, lastindex
import StatsBase: var, mean
import Statistics: cov 
using LinearAlgebra, SparseArrays, StatsBase, Distributions

# model.jl implements AbstractModel and SDEModel.
include("model.jl") #OK
# samplepath.jl implements AbstractSamplePath and SamplePath and
# extends Base.step, Base.length, to the appropriate parametric
# types.
include("samplepath.jl") #OK
# basisfunctions.jl implements the Fourier and Faber-Schauder basis.
include("basisfunctions.jl") #OK 
# SDE.jl implements AbstractSDE, SDE and extends
#Distribution.rand to sample an SDE sample path.
include("sde.jl") # OK
# GaussianVectors.jl implements Gaussian vectors in a flexible way.
include("gaussianvectors.jl") # OK 
# gaussianprocess.jl implements the abstract type AbstractGaussianProcess with
# subtypes GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
# Extends Base.length and Distributions.rand to AbstractGaussianProcess types.
include("gaussianprocess.jl")

# gaussianprocess.jl is in principe OK, maar nu nog var(Π::GaussianProcess) and
# cov(s,t, Π) toevoegen. Voeg support voor Distributions toe.

# calculateposterior.jl implements calculateposterior methods to calculate the
# posterior.
include("calculateposterior.jl")
