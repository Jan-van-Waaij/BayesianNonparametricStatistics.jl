import Base: length, step, rand
import StatsBase: var, mean
using LinearAlgebra, SparseArrays, StatsBase, Distributions

# model.jl implements AbstractModel and SDEModel.
include("model.jl")
# samplepath.jl implements AbstractSamplePath and SamplePath and
# extends Base.step, Base.length, to the appropriate parametric
# types.
include("samplepath.jl")
# basisfunctions.jl implements the Fourier and Faber-Schauder basis.
include("basisfunctions.jl")
# SDE.jl implements AbstractSDE, SDE and extends
#Distribution.rand to sample an SDE sample path.
include("sde.jl")
# GaussianVectors.jl implements Gaussian vectors in a flexible way.
include("gaussianvectors.jl")
# gaussianprocess.jl implements the abstract type AbstractGaussianProcess with
# subtypes GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
# Extends Base.length and Distributions.rand to AbstractGaussianProcess types.
include("gaussianprocess.jl")
# calculateposterior.jl implements calculateposterior methods to calculate the
# posterior.
include("calculateposterior.jl")
