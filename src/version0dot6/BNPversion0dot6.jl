import Base: length, step, mean, rand, var

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

#samplepath.jl are the same in both versions.
#basisfunctions.jl are the same in both versions.
#sde.jl are the same in both versions.
#gaussianvectors.jl are the same in both versions.
#gaussianprocess.jl are the same in both versions.
#calculateposterior.jl are the same in both versions.